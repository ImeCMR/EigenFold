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
parser.add_argument('--noesy_input_path', type=str, default=None, help="Path to the NOESY data file. Format: resFrom resTo peakID distance atomFrom atomTo")

inf_args = parser.parse_args()

import os, yaml, torch, wandb
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils.logging import get_logger
logger = get_logger(__name__)
from utils.dataset import get_loader
import pandas as pd
from model import get_model
from utils.inference import inference_epoch

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

def load_and_parse_noesy_data(noesy_file_path):
    """
    Loads and parses NOESY data from a file.
    Converts residue indices from 1-based to 0-based.
    """
    if not noesy_file_path or not os.path.exists(noesy_file_path):
        logger.info("NOESY input path not provided or file does not exist. Proceeding without NOESY data.")
        return None
    
    parsed_noesy_data = []
    try:
        with open(noesy_file_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 6:
                    logger.warning(f"Skipping malformed NOESY data line {line_num+1} in {noesy_file_path}: {line}. Expected 6 parts.")
                    continue

                try:
                    # Convert residue indices from 1-based to 0-based
                    res_from = int(parts[0]) - 1
                    res_to = int(parts[1]) - 1
                    peak_id = int(parts[2]) # peakID can remain as is
                    distance = float(parts[3])
                    atom_from = parts[4]
                    atom_to = parts[5]

                    if res_from < 0 or res_to < 0:
                        logger.warning(f"Skipping NOESY data line {line_num+1} with non-positive residue index after 0-based conversion: {line}")
                        continue

                    parsed_noesy_data.append([res_from, res_to, peak_id, distance, atom_from, atom_to])
                except ValueError as e:
                    logger.warning(f"Skipping NOESY data line {line_num+1} due to parsing error ({e}): {line}")
                    continue
        logger.info(f"Successfully loaded and parsed {len(parsed_noesy_data)} NOESY contacts from {noesy_file_path}.")
        if not parsed_noesy_data: # If file was empty or all lines were skipped
            return None
        return parsed_noesy_data
    except Exception as e:
        logger.error(f"Failed to read or parse NOESY file {noesy_file_path}: {e}")
        return None

def main():
    
    # Load NOESY data if provided
    noesy_data = load_and_parse_noesy_data(inf_args.noesy_input_path)

    logger.info(f'Loading splits {args.splits}')
    try: splits = pd.read_csv(args.splits).set_index('path')   
    except: splits = pd.read_csv(args.splits).set_index('name')   
    
    logger.info("Constructing model")
    model = get_model(args).to(device)
    ckpt = os.path.join(inf_args.model_dir, inf_args.ckpt)
    
    logger.info(f'Loading weights from {ckpt}')
    state_dict = torch.load(ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'], strict=True)
    ep = state_dict['epoch']
    
    val_loader = get_loader(args, None, splits, mode=inf_args.split_key, shuffle=False)
    # Pass noesy_data to inference_epoch.
    # Note: inference_epoch will need to be modified to accept and use this.
    # If NOESY data is specific per item in dataset, this global noesy_data variable won't directly work.
    # For now, we assume inference_epoch can handle a global NOESY data list for the items it processes,
    # or this script is intended for single PDB inference where noesy_data applies to that PDB.
    samples, log = inference_epoch(args, model, val_loader.dataset, device=device, pdbs=True, elbo=inf_args.elbo, noesy_data=noesy_data)
    
    means = {key: np.mean(log[key]) for key in log if key != 'path'}
    logger.info(f"Inference epoch {ep}: len {len(log['rmsd'])} MEANS {means}")
    
    inf_name = f"{args.splits.split('/')[-1]}.ep{ep}.num{args.num_samples}.step{args.inf_step}.alpha{args.alpha}.beta{args.beta}"
    if inf_args.inf_step != inf_args.elbo_step: inf_name += f".elbo{args.elbo_step}"
    csv_path = os.path.join(inf_args.model_dir, f'{inf_name}.csv')
    pd.DataFrame(log).set_index('path').to_csv(csv_path)
    logger.info(f"Saved inf csv {csv_path}")
    
    if not os.path.exists(os.path.join(inf_args.model_dir, inf_name)): os.mkdir(os.path.join(inf_args.model_dir, inf_name))
    for samp in samples:
        samp.pdb.write(os.path.join(inf_args.model_dir, inf_name, samp.path.split('/')[-1] + f".{samp.copy}.anim.pdb"), reverse=True)
        samp.pdb.clear().add(samp.Y).write(
            os.path.join(inf_args.model_dir, inf_name, samp.path.split('/')[-1] + f".{samp.copy}.pdb"))
        
if __name__ == '__main__':
    main()