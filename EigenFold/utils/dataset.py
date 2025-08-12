from functools import lru_cache
import numpy as np
import torch, os
from diffusion.sampling import ForwardDiffusionKernel
from diffusion import PolymerSDE
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
from .logging import get_logger
logger = get_logger(__name__)
from .pdb import pdb_to_npy

class ResidueDataset(Dataset):
    def __init__(self, args, split, **kwargs):
        super(ResidueDataset, self).__init__(**kwargs)
        self.split = split
        
        embeddings_arg_keys = ['omegafold_num_recycling']
        embeddings_suffix = get_args_suffix(embeddings_arg_keys, args) + '.npz'
        self.embeddings_suffix = embeddings_suffix
        
        self.lm_edge_dim = args.lm_edge_dim
        self.lm_node_dim = args.lm_node_dim
        self.args = args
        
    @lru_cache(maxsize=None)
    def get_sde(self, i):
        args = self.args
        sde = PolymerSDE(N=i, a=args.sde_a, b=args.sde_b)
        sde.make_schedule(Hf=args.train_Hf, step=args.inf_step, tmin=args.train_tmin)
        return sde
        
    def len(self):
        return len(self.split)
    
    def null_data(self, data):
        data.skip = True; return data
        
    
    def get(self, idx):
        if type(self.split) == list:
            subdf = self.split[idx][1]
            row = subdf.loc[np.random.choice(subdf.index)]
        else:
            row = self.split.iloc[idx]       
    
        data = HeteroData(); data.skip = False
        data['resi'].num_nodes = row.seqlen
        data['resi'].edge_index = get_dense_edges(row.seqlen)
        data.resi_sde = data.sde = self.get_sde(row.seqlen)
        data.path = pdb_path = os.path.join(self.args.pdb_dir, row.name[:2], row.name); data.info = row
        
        
        ret = pdb_to_npy(pdb_path, seqres=row.seqres)
        if not self.args.inference_mode and ret is None:
            logger.warning(f"Error loading {pdb_path}")
            return self.null_data(data)
        elif ret is not None:
            pos, mask = ret
            pos[~mask,0] = data.sde.conditional(mask, pos[mask,0])
            data['resi'].pos = torch.tensor(pos[:,0]).float()

            # --- Start of NOESY simulation logic ---
            # Simplified atom mapping for ILV heavy atoms based on a typical atom14 representation
            ILV_ATOM_IDX = {
                'I': [4, 6, 7, 9],   # CB, CG1, CG2, CD1
                'L': [4, 5, 9, 10],  # CB, CG, CD1, CD2
                'V': [4, 6, 7]       # CB, CG1, CG2
            }

            noesy_peaks = []
            try:
                ilv_residues = [(i, aa) for i, aa in enumerate(row.seqres) if aa in 'ILV']
                if len(ilv_residues) > 1:
                    all_ilv_atoms = []  # List of (res_idx, atom_idx, coords)
                    for res_idx, res_type in ilv_residues:
                        if not mask[res_idx]: continue
                        for atom_idx in ILV_ATOM_IDX[res_type]:
                            if not np.isnan(pos[res_idx, atom_idx, 0]):
                                all_ilv_atoms.append((res_idx, atom_idx, pos[res_idx, atom_idx]))

                    if len(all_ilv_atoms) > 1:
                        atom_identifiers = np.array([[ident[0], ident[1]] for ident in all_ilv_atoms])
                        atom_coords = np.array([ident[2] for ident in all_ilv_atoms])
                        dist_matrix = np.linalg.norm(atom_coords[:, None, :] - atom_coords[None, :, :], axis=-1)

                        true_peak_indices = np.argwhere((dist_matrix > 0) & (dist_matrix < 6.0))
                        true_peaks = []
                        for i, j in true_peak_indices:
                            if i >= j: continue
                            res_i, atom_i_idx = atom_identifiers[i]
                            res_j, atom_j_idx = atom_identifiers[j]
                            dist = dist_matrix[i, j]
                            true_peaks.append((res_i, res_j, dist, atom_i_idx, atom_j_idx, 1.0))

                        false_peak_indices = np.argwhere(dist_matrix > 10.0)
                        false_peaks = []
                        num_false_to_generate = int(len(true_peaks) * 0.1) # 10% of true peaks

                        if len(false_peak_indices) > num_false_to_generate > 0:
                            selected_indices = np.random.choice(len(false_peak_indices), num_false_to_generate, replace=False)
                            for k in selected_indices:
                                i, j = false_peak_indices[k]
                                if i >= j: continue
                                res_i, atom_i_idx = atom_identifiers[i]
                                res_j, atom_j_idx = atom_identifiers[j]
                                fake_dist = np.random.uniform(2.0, 5.0)
                                false_peaks.append((res_i, res_j, fake_dist, atom_i_idx, atom_j_idx, 0.0))

                        noesy_peaks = true_peaks + false_peaks
            except Exception as e:
                logger.warning(f"Could not simulate NOESY for {pdb_path} due to {e}")
                noesy_peaks = []

            if noesy_peaks:
                data.noesy_peaks = torch.tensor(noesy_peaks, dtype=torch.float32)
            else:
                data.noesy_peaks = torch.empty(0, 6, dtype=torch.float32)
            # --- End of NOESY simulation logic ---
        
        embeddings_name = row.__getattr__(self.args.embeddings_key)
        embeddings_path = os.path.join(self.args.embeddings_dir, embeddings_name[:2], embeddings_name) + '.' + self.embeddings_suffix
        if not os.path.exists(embeddings_path):
            logger.warning(f"No LM embeddings at {embeddings_path}")
            return self.null_data(data)
            
        try:
            embeddings_dict = dict(np.load(embeddings_path)) # {"node_repr": ..., "edge_repr": ...}
            node_repr, edge_repr = embeddings_dict['node_repr'], embeddings_dict['edge_repr']
        except:
            logger.warning(f"Error loading {embeddings_path}")
            return self.null_data(data)
        
        if self.args.no_edge_embs:
            edge_repr = np.zeros_like(edge_repr)
        
        if node_repr.shape[0] != data['resi'].num_nodes:
            logger.warning(f"LM dim error at {embeddings_path}: expected {data['resi'].num_nodes} got {node_repr.shape} {edge_repr.shape}")
            return self.null_data(data)
            
        data['resi'].node_attr = torch.tensor(node_repr)
        edge_repr = torch.tensor(edge_repr)
        src, dst = data['resi'].edge_index[0], data['resi'].edge_index[1]
        data['resi'].edge_attr_ = torch.cat([edge_repr[src, dst], edge_repr[dst, src]], -1)
        
        return data
    
def get_loader(args, pyg_data, splits, mode='train', shuffle=True):
   
    if 'sde_a' not in args.__dict__:
        args.sde_a = 3/(3.8**2)
    if 'sde_b' not in args.__dict__:
        args.sde_b = 0
    if 'no_edge_embs' not in args.__dict__:
        args.no_edge_embs = False
   
    try:
        split = splits[splits.split == mode]
    except:
        split = splits
        logger.warning("Not splitting based on split")
    
    
    if args.limit_mols:
        split = split[:args.limit_mols]
    if 'seqlen' not in split.columns:
        split['seqlen'] = [len(s) for s in split.seqres]
    split = split[split.seqlen <= args.max_len]
    
    transform = ForwardDiffusionKernel(args)
    dataset = ResidueDataset(args, split, transform=transform)
        
    logger.info(f"Initialized {mode if mode else ''} loader with {len(dataset)} entries")
    loader = DataLoader(dataset=dataset,
        batch_size=args.batch, shuffle=shuffle, pin_memory=False,
        num_workers=args.num_workers
    )
    
    return loader



def get_args_suffix(arg_keys, args):
    cache_name = []
    for k in arg_keys: cache_name.extend([k, args.__dict__[k]])
    return '.'.join(map(str, cache_name))

def get_dense_edges(n):
    atom_ids = np.arange(n)
    src, dst = np.repeat(atom_ids, n), np.tile(atom_ids, n)
    mask = src != dst; src, dst = src[mask], dst[mask]
    edge_idx = np.stack([src, dst])
    return torch.tensor(edge_idx)
