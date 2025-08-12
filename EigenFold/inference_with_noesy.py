import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True)

parser.add_argument('--splits', type=str, required=True)
parser.add_argument('--split_key', type=str, default=None)
parser.add_argument('--inf_mols', type=int, default=1000)
parser.add_argument('--wandb', type=str, default=None)
parser.add_argument('--num_workers', type=int, default=None)

parser.add_argument('--ode', action='store_true', default=False)
parser.add_argument('--elbo', action='store_true', default=False)
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--num_samples', type=int, default=1)

parser.add_argument('--inf_step', type=float, default=0.5)
parser.add_argument('--elbo_step', type=float, default=0.2)
parser.add_argument('--inf_type', type=str,
                        choices=['entropy', 'rate'], default='rate')
parser.add_argument('--max_len', type=int, default=1024)

parser.add_argument('--inf_Hf', type=float, default=None)
parser.add_argument('--inf_kmin', type=int, default=None)
parser.add_argument('--inf_tmin', type=int, default=None)
parser.add_argument('--inf_cutoff', type=int, default=None)

parser.add_argument('--embeddings_dir', type=str, default=None)
parser.add_argument('--pdb_dir', type=str, default=None)
parser.add_argument('--embeddings_key', type=str, default=None, choices=['name', 'reference'])
parser.add_argument('--noesy_file', type=str, required=True, help='Path to the NOESY peak list file.')

inf_args = parser.parse_args()

import os, yaml, torch, wandb
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils.logging import get_logger
logger = get_logger(__name__)
from utils.dataset import get_dense_edges, get_args_suffix
from diffusion import PolymerSDE
import pandas as pd
from model import get_model
from utils.inference import inference_epoch
from torch_geometric.data import HeteroData

def parse_noesy_file(path):
    """Parses a NOESY file into a tensor."""
    # This is a simplified mapping. A more robust solution would be needed for general use.
    # It only maps a few heavy atoms as a proxy for their protons.
    ATOM_NAME_TO_IDX = {
        'CB': 4, 'CG': 5, 'CG1': 6, 'CG2': 7, 'CD': 8, 'CD1': 9, 'CD2': 10
    }
    peaks = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            try:
                res_i = int(parts[0]) - 1 # 1-based to 0-based
                res_j = int(parts[1]) - 1 # 1-based to 0-based
                dist = float(parts[2])
                atom_i_name = parts[3].upper()
                atom_j_name = parts[4].upper()

                # For simplicity, we map common side-chain heavy atoms.
                # A real implementation would need a full proton mapping.
                atom_i_idx = ATOM_NAME_TO_IDX.get(atom_i_name, -1)
                atom_j_idx = ATOM_NAME_TO_IDX.get(atom_j_name, -1)

                if atom_i_idx != -1 and atom_j_idx != -1:
                    # Add placeholder for the true/false label, which is unknown at inference.
                    # Using -1 as a clear indicator that this is inference data.
                    peaks.append((res_i, res_j, dist, atom_i_idx, atom_j_idx, -1.0))
            except (ValueError, IndexError):
                logger.warning(f"Could not parse NOESY line: {line.strip()}")
                continue
    if not peaks:
        return torch.empty(0, 6, dtype=torch.float32)
    return torch.tensor(peaks, dtype=torch.float32)


with open(f'{inf_args.model_dir}/args.yaml') as f:
    args = argparse.Namespace(**yaml.full_load(f))
args.wandb = inf_args.wandb
if inf_args.wandb:
    wandb.login(key=os.environ['WANDB_API_KEY'])
    wandb.init(
        entity=os.environ['WANDB_ENTITY'],
        settings=wandb.Settings(start_method="fork"),
        project=args.wandb,
        name=str(args.time),
        config=args.__dict__ | inf_args.__dict__
    )
args.splits = inf_args.splits
args.inf_mols = inf_args.inf_mols
args.inf_type = inf_args.inf_type
args.num_samples = inf_args.num_samples
args.inf_step = inf_args.inf_step
args.max_len = inf_args.max_len
args.ode = inf_args.ode
args.alpha, args.beta = inf_args.alpha, inf_args.beta
args.elbo_step = inf_args.elbo_step

if inf_args.num_workers is not None:
    args.num_workers = inf_args.num_workers

if inf_args.inf_Hf:
    if inf_args.inf_Hf < args.train_Hf: logger.warning(f'Out of bounds: inf_Hf {inf_args.inf_Hf} < train_Hf {args.train_Hf}')
    args.train_Hf = inf_args.inf_Hf

if inf_args.inf_kmin:
    if inf_args.inf_kmin < args.train_kmin: logger.warning(f'Out of bounds: inf_kmin {inf_args.inf_kmin} < train_kmin {args.train_kmin}')
    args.train_kmin = inf_args.inf_kmin

if inf_args.inf_tmin:
    if inf_args.inf_tmin < args.train_tmin: logger.warning(f'Out of bounds: inf_tmin {inf_args.inf_tmin} < train_tmin {args.train_tmin}')
    args.train_tmin = inf_args.inf_tmin

if inf_args.inf_cutoff:
    if inf_args.inf_cutoff > args.train_cutoff: logger.warning(f'Out of bounds: inf_cutoff {inf_args.inf_cutoff} > train_cutoff {args.train_cutoff}')

if inf_args.pdb_dir: args.pdb_dir = inf_args.pdb_dir
if inf_args.embeddings_dir: args.embeddings_dir = inf_args.embeddings_dir
if inf_args.embeddings_key: args.embeddings_key = inf_args.embeddings_key
args.inference_mode = True

def main():

    logger.info(f"Loading sequence info from {args.splits}")
    # We expect a CSV with one row for the protein of interest
    # It must contain 'name' and 'seqres' columns
    try:
        protein_info = pd.read_csv(args.splits).iloc[0]
    except Exception as e:
        logger.error(f"Could not read the splits file {args.splits}. It should be a CSV with 'name' and 'seqres' columns. Error: {e}")
        return

    logger.info(f"Loading NOESY data from {inf_args.noesy_file}")
    noesy_data = parse_noesy_file(inf_args.noesy_file)
    if noesy_data.shape[0] == 0:
        logger.warning("No valid peaks found in the NOESY file.")

    # Manually construct the data object for a single protein
    data = HeteroData()
    data.skip = False
    data.info = protein_info
    data.path = protein_info.name

    seqlen = len(protein_info.seqres)
    data['resi'].num_nodes = seqlen
    data['resi'].edge_index = get_dense_edges(seqlen)
    data.sde = data.resi_sde = PolymerSDE(N=seqlen, a=args.sde_a, b=args.sde_b)
    data.sde.make_schedule(Hf=args.train_Hf, step=args.inf_step, tmin=args.train_tmin)

    # Load pre-computed embeddings
    embeddings_arg_keys = ['omegafold_num_recycling']
    embeddings_suffix = get_args_suffix(embeddings_arg_keys, args) + '.npz'
    embeddings_name = protein_info.__getattr__(args.embeddings_key)
    embeddings_path = os.path.join(args.embeddings_dir, embeddings_name[:2], embeddings_name) + '.' + embeddings_suffix
    if not os.path.exists(embeddings_path):
        logger.error(f"Could not find embeddings at {embeddings_path}. Please run make_embeddings.py first.")
        return

    embeddings_dict = dict(np.load(embeddings_path))
    node_repr, edge_repr = embeddings_dict['node_repr'], embeddings_dict['edge_repr']
    data['resi'].node_attr = torch.tensor(node_repr)
    src, dst = data['resi'].edge_index[0], data['resi'].edge_index[1]
    data['resi'].edge_attr_ = torch.cat([torch.tensor(edge_repr)[src, dst], torch.tensor(edge_repr)[dst, src]], -1)

    # Attach parsed NOESY data
    data.noesy_peaks = noesy_data.to(device)

    # Load Model
    logger.info("Constructing model")
    model = get_model(args).to(device)
    ckpt = os.path.join(inf_args.model_dir, inf_args.ckpt)
    logger.info(f'Loading weights from {ckpt}')
    state_dict = torch.load(ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'], strict=True)
    ep = state_dict['epoch']

    # Run Inference
    dataset = [data] # The inference_epoch function expects a list of data objects
    samples, log = inference_epoch(args, model, dataset, device=device, pdbs=True, elbo=inf_args.elbo)

    # Save Results
    means = {key: np.mean(log[key]) for key in log if key != 'path'}
    logger.info(f"Inference epoch {ep}: len {len(log['rmsd'])} MEANS {means}")

    inf_name = f"{protein_info.name}.noesy.ep{ep}.num{args.num_samples}.step{args.inf_step}"
    csv_path = os.path.join(inf_args.model_dir, f'{inf_name}.csv')
    pd.DataFrame(log).set_index('path').to_csv(csv_path)
    logger.info(f"Saved inf csv {csv_path}")

    output_dir = os.path.join(inf_args.model_dir, inf_name)
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for samp in samples:
        output_pdb_name = samp.path.split('/')[-1]
        samp.pdb.write(os.path.join(output_dir, output_pdb_name + f".{samp.copy}.anim.pdb"), reverse=True)
        samp.pdb.clear().add(samp.Y).write(
            os.path.join(output_dir, output_pdb_name + f".{samp.copy}.pdb"))
    logger.info(f"Saved PDB predictions to {output_dir}")

if __name__ == '__main__':
    main()
