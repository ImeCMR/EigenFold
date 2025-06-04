from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph, knn_graph, knn
from torch_scatter import scatter, scatter_mean, scatter_max
import numpy as np
from e3nn.nn import BatchNorm
from .utils import get_timestep_embedding, GaussianSmearing, sinusoidal_embedding

class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, in_tp_irreps, out_tp_irreps,
                 sh_irreps, out_irreps, n_edge_features,
                 batch_norm=False, dropout=0.0, node_feature_dim=4,
                 fc_dim=32, lin_self=False, attention=False):
        super(TensorProductConvLayer, self).__init__()
        
        # consider attention...
        self.nf = nf = node_feature_dim
        self.lin_in = o3.Linear(in_irreps, in_tp_irreps, internal_weights=True)
        self.tp = tp = o3.FullyConnectedTensorProduct(in_tp_irreps, sh_irreps, out_tp_irreps, shared_weights=False)      
        self.lin_out = o3.Linear(out_tp_irreps, out_irreps, internal_weights=True)
        if lin_self:
            self.lin_self = o3.Linear(in_irreps, out_irreps, internal_weights=True)
        else: self.lin_self = False
        if attention:
            self.attention = True
            key_irreps = [(mul//2, ir) for mul, ir in in_tp_irreps]
            self.h_q = o3.Linear(in_tp_irreps, key_irreps)
            self.tp_k = tp_k = o3.FullyConnectedTensorProduct(in_tp_irreps, sh_irreps, key_irreps, shared_weights=False)
            self.fc_k = self.fc = nn.Sequential(
                nn.Linear(n_edge_features, fc_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_dim, fc_dim),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_dim, tp_k.weight_numel)
            )
            self.dot = o3.FullyConnectedTensorProduct(key_irreps, key_irreps, "0e")
        else: self.attention = False
        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, fc_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_dim, fc_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_dim, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    
    def forward(self, node_attr, edge_index, edge_attr, edge_sh, ones=None, residual=True, out_nodes=None, reduce='mean'):
        node_attr_in = self.lin_in(node_attr)
        edge_src, edge_dst = edge_index
        out_nodes = out_nodes or node_attr.shape[0]
        if self.attention:
            def ckpt_forward(node_attr_in, edge_src, edge_dst, edge_sh, edge_attr):
                q = self.h_q(node_attr_in)
                k = self.tp_k(node_attr_in[edge_src], edge_sh, self.fc_k(edge_attr))
                v = self.tp(node_attr_in[edge_src], edge_sh, self.fc(edge_attr))
                a = self.dot(q[edge_dst], k)
                max_ = scatter_max(a, edge_dst, dim=0, dim_size=out_nodes)[0]
                a = (a - max_[edge_dst]).exp()
                z = scatter(a, edge_dst, dim=0, dim_size=out_nodes)
                a = a / z[edge_dst]
                return scatter(a * v, edge_dst, dim=0, dim_size=out_nodes)
        else:
            def ckpt_forward(node_attr_in, edge_src, edge_dst, edge_sh, edge_attr):
                tp = self.tp(node_attr_in[edge_src], edge_sh, self.fc(edge_attr))
                return scatter(tp, edge_dst, dim=0, dim_size=out_nodes, reduce=reduce)
        if self.training:        
            out = torch.utils.checkpoint.checkpoint(ckpt_forward,
                    node_attr_in, edge_src, edge_dst, edge_sh, edge_attr)
        else:
            out = ckpt_forward(node_attr_in, edge_src, edge_dst, edge_sh, edge_attr)
        
        out = self.lin_out(out)
        
        if not residual:
            return out
        if self.lin_self: 
            out = out + self.lin_self(node_attr)
        else:
            out = out + F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1])) 
        if self.batch_norm:
            out = self.batch_norm(out)
        return out

class ResiLevelTensorProductScoreModel(torch.nn.Module):
    def __init__(self, args):
        super(ResiLevelTensorProductScoreModel, self).__init__()
        
        self.args = args
        self.t_emb_func = get_timestep_embedding(args.t_emb_type, args.t_emb_dim)
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=args.sh_lmax)
        #self.normalize_pred = args.normalize_pred
        
        lm_node_dim = args.lm_node_dim
        self.resi_node_embedding = nn.Sequential(
            nn.Linear(args.t_emb_dim + lm_node_dim, args.resi_ns), # fix
            nn.ReLU(),
            nn.Linear(args.resi_ns, args.resi_ns),
            nn.ReLU(),
            nn.Linear(args.resi_ns, args.resi_ns)
        )
        lm_edge_dim = args.lm_edge_dim 
        self.resi_edge_embedding = nn.Sequential(
            nn.Linear(args.t_emb_dim + args.radius_emb_dim + args.resi_pos_emb_dim + 2*lm_edge_dim, args.resi_ns), # fix
            nn.ReLU(),
            nn.Linear(args.resi_ns, args.resi_ns),
            nn.ReLU(),
            nn.Linear(args.resi_ns, args.resi_ns)
        )
        self.resi_node_norm = nn.LayerNorm(lm_node_dim)
        self.resi_edge_norm = nn.LayerNorm(2*lm_edge_dim)
        if args.no_radius_sqrt:
            if args.radius_emb_type == 'gaussian':
                self.distance_expansion = GaussianSmearing(0.0, args.radius_emb_max, args.radius_emb_dim)
            elif args.radius_emb_type == 'sinusoidal':
                self.distance_expansion = lambda x: sinusoidal_embedding(10000 * x/args.radius_emb_max, args.radius_emb_dim)
        else:
            if args.radius_emb_type == 'gaussian':
                self.distance_expansion = GaussianSmearing(0.0, args.radius_emb_max**0.5, args.radius_emb_dim)
            elif args.radius_emb_type == 'sinusoidal':
                self.distance_expansion = lambda x: sinusoidal_embedding(10000 * x/args.radius_emb_max**0.5, args.radius_emb_dim)
        
        conv_layers = []
        if args.order == 2:
            irrep_seq = [
                [(0, 1)],
                [(0, 1), (1, -1), (2, 1)],
                [(0, 1), (1, -1), (2, 1), (1, 1), (2, -1)],
                [(0, 1), (1, -1), (2, 1), (1, 1), (2, -1), (0, -1)],
            ]
        else:
            irrep_seq = [
                [(0, 1)],
                [(0, 1), (1, -1)],
                [(0, 1), (1, -1), (1, 1)],
                [(0, 1), (1, -1), (1, 1), (0, -1)]
            ]
        def fill_mults(ns, nv, irs, is_in=False):
            irreps = [(ns, (l, p)) if (l == 0 and p == 1) else [nv, (l, p)] for l, p in irs]
            return irreps
        for i in range(args.resi_conv_layers):
            ns, nv, ntps, ntpv, fc_dim = args.resi_ns, args.resi_nv, args.resi_ntps, args.resi_ntpv, args.resi_fc_dim
            in_seq, out_seq = irrep_seq[min(i, len(irrep_seq) - 1)], irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            in_irreps = fill_mults(ns, nv, in_seq, is_in=(i==0))
            out_irreps = fill_mults(ns, nv, out_seq)
            in_tp_irreps = fill_mults(ntps, ntpv, in_seq)
            out_tp_irreps = fill_mults(ntps, ntpv, out_seq)
            layer = TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                in_tp_irreps=in_tp_irreps,
                out_tp_irreps=out_tp_irreps,
                out_irreps=out_irreps,
                n_edge_features=3*ns,
                batch_norm=False,
                node_feature_dim=min(ns, args.lin_nf),
                fc_dim=fc_dim,
                lin_self=args.lin_self,
                attention=args.attention
            )
            conv_layers.append(layer)
                
        self.conv_layers = nn.ModuleList(conv_layers)
        self.resi_final_tp = o3.FullyConnectedTensorProduct(out_irreps, out_irreps, '1x1o + 1x1e' if args.parity else '1x1o', internal_weights=True)

        # --- NOESY Specific Initializations ---
        self.casp13_atom_map = {
            'H': 0, 'HB': 1, 'HG1': 2, 'HG2': 3, 'HD1': 4, 'HD2': 5, 'HE2': 6,
            # Add other relevant CASP13 atom types as needed
        } # TODO: Make this more comprehensive or pass as arg
        self.num_noesy_atom_types = len(self.casp13_atom_map)

        # TODO: Make these configurable via args
        self.noesy_emb_dim = args.noesy_emb_dim if hasattr(args, 'noesy_emb_dim') else 32  # Embedding dim for NOESY distance
        self.noesy_atom_emb_dim = args.noesy_atom_emb_dim if hasattr(args, 'noesy_atom_emb_dim') else 16 # Embedding dim for atom types
        self.noesy_feature_dim = args.noesy_feature_dim if hasattr(args, 'noesy_feature_dim') else 64 # Final NOESY feature dim per contact

        self.noesy_distance_expansion = GaussianSmearing(0.0, 10.0, self.noesy_emb_dim) # Assuming NOESY distances are within 0-10 Angstroms
        self.noesy_atom_from_embedding = nn.Embedding(self.num_noesy_atom_types, self.noesy_atom_emb_dim)
        self.noesy_atom_to_embedding = nn.Embedding(self.num_noesy_atom_types, self.noesy_atom_emb_dim)
        
        # Linear layer to project concatenated NOESY features (dist_emb + atom_from_emb + atom_to_emb)
        self.noesy_feature_projection = nn.Linear(self.noesy_emb_dim + 2 * self.noesy_atom_emb_dim, self.noesy_feature_dim)

        # Update resi_edge_embedding input dimension
        # Original: args.t_emb_dim + args.radius_emb_dim + args.resi_pos_emb_dim + 2*lm_edge_dim
        # New: original + self.noesy_feature_dim
        original_edge_dim = args.t_emb_dim + args.radius_emb_dim + args.resi_pos_emb_dim + 2*lm_edge_dim
        self.resi_edge_embedding = nn.Sequential(
            nn.Linear(original_edge_dim + self.noesy_feature_dim, args.resi_ns),
            nn.ReLU(),
            nn.Linear(args.resi_ns, args.resi_ns),
            nn.ReLU(),
            nn.Linear(args.resi_ns, args.resi_ns)
        )
        # --- End NOESY Specific Initializations ---

    def _get_atom_idx(self, atom_name):
        return self.casp13_atom_map.get(atom_name, self.casp13_atom_map.get('H')) # Default to 'H' if not found

    def _process_noesy_data(self, noesy_data, num_residues, device):
        """
        Processes raw NOESY data into a feature tensor for edge augmentation.
        noesy_data: list of [res_idx_from, res_idx_to, peak_id, distance, atom_name_from, atom_name_to]
        Returns a sparse tensor representing NOESY features on a (num_residues, num_residues) grid.
        """
        if noesy_data is None or len(noesy_data) == 0:
            # Return a zero tensor of appropriate shape for concatenation if no NOESY data
            return torch.zeros((num_residues, num_residues, self.noesy_feature_dim), device=device)

        # Assuming noesy_data is a list of lists/tuples on CPU, convert to tensor
        # Columns: res_idx_from, res_idx_to, peak_id, distance, atom_name_from, atom_name_to
        # We need: res_idx_from, res_idx_to, distance, atom_idx_from, atom_idx_to

        res_indices_from = torch.tensor([int(n[0]) for n in noesy_data], dtype=torch.long, device=device)
        res_indices_to = torch.tensor([int(n[1]) for n in noesy_data], dtype=torch.long, device=device)
        distances = torch.tensor([float(n[3]) for n in noesy_data], dtype=torch.float, device=device)
        atom_indices_from = torch.tensor([self._get_atom_idx(n[4]) for n in noesy_data], dtype=torch.long, device=device)
        atom_indices_to = torch.tensor([self._get_atom_idx(n[5]) for n in noesy_data], dtype=torch.long, device=device)

        dist_emb = self.noesy_distance_expansion(distances)
        atom_from_emb = self.noesy_atom_from_embedding(atom_indices_from)
        atom_to_emb = self.noesy_atom_to_embedding(atom_indices_to)

        concatenated_features = torch.cat([dist_emb, atom_from_emb, atom_to_emb], dim=-1)
        projected_features = self.noesy_feature_projection(concatenated_features) # (num_contacts, noesy_feature_dim)

        # Create a sparse representation for NOESY features on a (N, N) residue grid
        # This allows summing features if multiple NOESY contacts exist for the same residue pair
        noesy_edge_features_sparse = torch.sparse_coo_tensor(
            indices=torch.stack([res_indices_from, res_indices_to]),
            values=projected_features,
            size=(num_residues, num_residues, self.noesy_feature_dim),
            device=device
        )
        # Convert to dense. This might be memory intensive for very large numbers of residues.
        # Consider alternatives if num_residues is very large.
        return noesy_edge_features_sparse.to_dense()


    def forward(self, data, noesy_data=None, **kwargs): # Added noesy_data
        
        data['resi'].x = self.resi_node_norm(data['resi'].node_attr)
        data['resi'].edge_attr = self.resi_edge_norm(data['resi'].edge_attr_) # problem
            
        
        ### BUILD RESI CONV GRAPH
        node_attr, edge_index, edge_attr, edge_sh = self.build_conv_graph(data, key='resi', knn=False, edge_pos_emb=True)

        # --- Process and Integrate NOESY Data ---
        num_residues = data['resi'].x.shape[0] # Number of nodes (residues)
        device = data['resi'].x.device

        processed_noesy_features_matrix = self._process_noesy_data(noesy_data, num_residues, device)

        # Augment edge_attr with NOESY features
        # edge_index gives (src, dst) pairs. Use these to lookup features in processed_noesy_features_matrix.
        src_nodes, dst_nodes = edge_index[0], edge_index[1]

        # Gather NOESY features for existing graph edges
        # For an edge (u, v), we can take features from processed_noesy_features_matrix[u, v]
        # and potentially also from processed_noesy_features_matrix[v, u] if NOESY data is symmetric or directional.
        # Assuming NOESY data implies undirected constraint, so sum/average features from (u,v) and (v,u) if they exist.
        # For simplicity, just take (u,v) for now.
        noesy_features_for_edges = processed_noesy_features_matrix[src_nodes, dst_nodes]

        # Concatenate NOESY features to the original edge_attr
        edge_attr = torch.cat([edge_attr, noesy_features_for_edges], dim=-1)
        # --- End NOESY Data Integration ---
        
        node_attr = self.resi_node_embedding(node_attr)
        edge_attr = self.resi_edge_embedding(edge_attr)
        
        src, dst = edge_index
        
        for layer in self.conv_layers:
            ns = self.args.resi_ns
            edge_attr_ = torch.cat([edge_attr, node_attr[src, :ns],
                                    node_attr[dst, :ns]], -1)
            node_attr = layer(node_attr, edge_index, edge_attr_, edge_sh)
    
        resi_out = self.resi_final_tp(node_attr, node_attr)
        if self.args.parity:
            resi_out = resi_out.view(-1, 2, 3).mean(1)    

        try: resi_out = resi_out * data.score_norm[:,None]
        except: resi_out = resi_out * data.score_norm
        
        data['resi'].pred = resi_out
        
        data.pred = resi_out
        return resi_out
    
    
    def build_conv_graph(self, data, key='atom', knn=True, edge_pos_emb=False):
        edge_index, edge_attr = data[key].edge_index, data[key].edge_attr
            
        node_t = torch.log(data[key].node_t / self.args.tmin) / np.log(self.args.tmax / self.args.tmin) * 10000
        node_t_emb = self.t_emb_func(node_t)
        node_attr = torch.cat([node_t_emb, data[key].x], 1)
        
        edge_t_emb = node_t_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_t_emb, edge_attr], 1)
        src, dst = edge_index
        if edge_pos_emb:
            edge_pos_emb = sinusoidal_embedding(dst - src, self.args.resi_pos_emb_dim)
            edge_attr = torch.cat([edge_pos_emb, edge_attr], 1)
        edge_vec = data[key].pos[src.long()] - data[key].pos[dst.long()]
        if self.args.no_radius_sqrt:
            edge_length_emb = self.distance_expansion(edge_vec.norm(dim=-1))
        else:
            edge_length_emb = self.distance_expansion(edge_vec.norm(dim=-1)**0.5)
        
        edge_attr = torch.cat([edge_length_emb, edge_attr], 1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component').float()
        return node_attr, edge_index, edge_attr, edge_sh
    
 