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
        self.use_noesy_data = args.use_noesy_data if hasattr(args, 'use_noesy_data') else False
        if self.use_noesy_data:
            logger.info("NOESY data usage is enabled for the dataset.")
        
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

        if self.use_noesy_data:
            # Attempt to load pre-generated NOESY data.
            # This assumes NOESY data is stored in a file named like the embedding file but with a '.noesy.pt' suffix.
            # e.g., if embedding is /path/to/embed.omegafold_num_recycling.3.npz
            # NOESY data would be /path/to/embed.noesy.pt
            noesy_data_path = embeddings_path.replace(self.embeddings_suffix, 'noesy.pt') # A bit of a guess for the naming scheme
            if os.path.exists(noesy_data_path):
                try:
                    loaded_noesy_data = torch.load(noesy_data_path)
                    # Ensure it's in the expected list-of-lists format, not a tensor from _process_noesy_data
                    if isinstance(loaded_noesy_data, list):
                        data.noesy_data = loaded_noesy_data
                        if idx < 5: # Log for a few samples to confirm loading
                            logger.info(f"Loaded {len(data.noesy_data)} NOESY contacts for {row.name} from {noesy_data_path}")
                    else:
                        logger.warning(f"NOESY data at {noesy_data_path} is not in expected list format. Skipping for {row.name}")
                        data.noesy_data = None
                except Exception as e:
                    logger.warning(f"Error loading NOESY data from {noesy_data_path} for {row.name}: {e}. Skipping NOESY for this item.")
                    data.noesy_data = None
            else:
                # Log only for the first few misses to avoid spamming
                if idx < 5 and not hasattr(self, '_logged_noesy_missing'): self._logged_noesy_missing = set()
                if idx < 5 and row.name not in self._logged_noesy_missing:
                    logger.info(f"No pre-generated NOESY data file found at {noesy_data_path} for {row.name}. Proceeding without NOESY for this item.")
                    self._logged_noesy_missing.add(row.name)
                data.noesy_data = None
        else:
            data.noesy_data = None

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
