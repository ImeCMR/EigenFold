import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=str, default='./data/pdb.dat')
parser.add_argument('--data', type=str, default='./data/pdb')
parser.add_argument('--outdir', type=str, default='./data/pdb_chains')
parser.add_argument('--outcsv', type=str, default='./data/pdb_chains.csv')
parser.add_argument('--num_workers', type=int, default=15)
args = parser.parse_args()

from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import Polypeptide
from Bio import SeqIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings, tqdm, os, io
from collections import defaultdict
import pandas as pd
import numpy as np
from multiprocessing import Pool
parser = PDBParser()

def main():
    with open(args.manifest) as f:
        manifest = [line for line in f.readlines() if line.strip()]
    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    infos = list(tqdm.tqdm(__map__(unpack_pdb, manifest), total=len(manifest)))
    if args.num_workers > 1:
        p.__exit__(None, None, None)

    # Flatten the list of lists into a single list of dictionaries
    info_list = [item for sublist in infos for item in sublist]

    if not info_list:
        print("No data processed. Exiting.")
        return

    df = pd.DataFrame.from_records(info_list, index='name')
    
    reps = []
    lookup = {}
    for seq, sub_df in tqdm.tqdm(df.groupby('seqres')):
        sub_df = sub_df.sort_values('release_date')
        rep = sub_df.index[0]
        reps.append(rep)
        for s in sub_df.index:
            lookup[s] = rep
    df['reference'] = [lookup[s] for s in df.index]
    
    df.to_csv(args.outcsv, index=True, index_label='name')
    
def unpack_pdb(pdb_id):
    in_path = os.path.join(args.data, pdb_id.strip())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        try:
            struct = parser.get_structure('', in_path)
            model = struct[0]
            header = struct.header
        except Exception as e:
            raise e
            if type(e) is KeyboardInterrupt: raise e
            return []
        
    seqres = {}
    for record in SeqIO.parse(in_path, "pdb-seqres"):
        seqres[record.annotations['chain']] = str(record.seq)

    infos = []
    for chain_id in model.child_dict:
        if chain_id not in seqres: continue
        chain = model.child_dict[chain_id]
        name = pdb_id[3:8] + chain_id + '.pdb'
        info = process_chain(chain, name)

        # Correctly sanitize data types before returning from the worker process
        for key in ['head', 'deposition_date', 'release_date', 'structure_method']:
            value = header.get(key)
            info[key] = str(value) if value is not None else ""

        res_val = header.get('resolution')
        info['resolution'] = float(res_val) if res_val is not None else np.nan

        info['seqres'] = seqres[chain_id]    
        infos.append(info)
    return infos
    
def process_chain(chain, name):
    info = {
        'name': name,
        'saved': False,
        'valid_alphas': 0,
        'seq': ''
    }
    try:
        for resi in list(chain):
            if (resi.id[0] != ' ') or ('CA' not in resi.child_dict):
                chain.detach_child(resi.id)            
        info['valid_alphas'] = int(len(chain)) # Cast to standard Python int
        info['seq'] = str(Polypeptide(chain).get_sequence())
        
        namedir = os.path.join(args.outdir, info['name'][:2])
        if not os.path.exists(namedir): os.makedirs(namedir, exist_ok=True)
        out_path = os.path.join(namedir, info['name'])

        pdbio = PDBIO()
        pdbio.set_structure(chain)
        pdbio.save(out_path, select=NotDisordered())
        info['saved'] = True
    except Exception as e:
        raise e
        pass
    return info
    
class NotDisordered(Select):
    def accept_atom(self, atom):
        return (not atom.is_disordered()) or (atom.get_altloc() == "A")


if __name__ == "__main__":
    main()