import argparse
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import Polypeptide
from Bio import SeqIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
import tqdm
import os
import io
import csv
from collections import defaultdict
import pandas as pd
import numpy as np
from multiprocessing import Pool

# Global parser and args so they are accessible by worker processes
parser = PDBParser()
args = None

class NotDisordered(Select):
    def accept_atom(self, atom):
        return (not atom.is_disordered()) or (atom.get_altloc() == "A")

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
        info['valid_alphas'] = int(len(chain))
        info['seq'] = str(Polypeptide(chain).get_sequence())

        namedir = os.path.join(args.outdir, info['name'][:2])
        if not os.path.exists(namedir): os.makedirs(namedir, exist_ok=True)
        out_path = os.path.join(namedir, info['name'])

        pdbio = PDBIO()
        pdbio.set_structure(chain)
        pdbio.save(out_path, select=NotDisordered())
        info['saved'] = True
    except Exception as e:
        # Log error but don't crash the whole pool
        print(f"Error processing chain {name}: {e}")
        pass
    return info

def unpack_pdb(pdb_id):
    in_path = os.path.join(args.data, pdb_id.strip() + '.pdb')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        try:
            struct = parser.get_structure('', in_path)
            model = struct[0]
            header = struct.header
        except Exception as e:
            print(f"Error parsing {in_path}: {e}")
            if type(e) is KeyboardInterrupt: raise e
            return []

    seqres = {}
    for record in SeqIO.parse(in_path, "pdb-seqres"):
        seqres[record.annotations['chain']] = str(record.seq)

    infos = []
    for chain_id in model.child_dict:
        chain = model.child_dict[chain_id]
        name = pdb_id.strip() + '.' + chain_id + '.pdb'
        info = process_chain(chain, name)

        # Correctly sanitize data types
        for key in ['head', 'deposition_date', 'release_date', 'structure_method']:
            value = header.get(key)
            info[key] = str(value) if value is not None else ""

        res_val = header.get('resolution')
        info['resolution'] = float(res_val) if res_val is not None else None

        if chain_id in seqres:
            info['seqres'] = seqres[chain_id]
        else:
            info['seqres'] = ''
        infos.append(info)
    return infos

def unpack_and_clean_pdbs():
    """
    Processes all PDBs from a manifest file in parallel.
    Saves cleaned, single-chain PDB files and returns a list of metadata dictionaries.
    """
    print("Stage 1: Unpacking and cleaning PDB files...")
    with open(args.manifest) as f:
        manifest = [line for line in f.readlines() if line.strip()]

    info_list = []
    if args.num_workers > 1:
        with Pool(args.num_workers) as p:
            # Use tqdm to show progress for imap
            with tqdm.tqdm(total=len(manifest)) as pbar:
                for infos in p.imap_unordered(unpack_pdb, manifest):
                    info_list.extend(infos)
                    pbar.update()
    else: # For debugging
        for pdb_id in tqdm.tqdm(manifest):
            info_list.extend(unpack_pdb(pdb_id))

    print(f"Finished processing. Found {len(info_list)} total chains.")
    return info_list

def create_master_dataframe(info_list):
    """
    Takes a list of info dicts, creates and sanitizes a DataFrame,
    adds the 'reference' column, and saves it to a CSV.
    """
    if not info_list:
        print("No chain information to process. Exiting.")
        return None

    print("Creating master DataFrame...")
    # Use from_records for robust creation
    df = pd.DataFrame.from_records(info_list, index='name')

    # Manual de-duplication to create the 'reference' column
    print("Grouping chains by sequence to find representatives...")
    seq_groups = defaultdict(list)
    # We need to iterate through the DataFrame to use its indexing
    for name, row in df.iterrows():
        d = row.to_dict()
        d['name'] = name
        seq_groups[row['seqres']].append(d)

    lookup = {}
    for seq, info_group in tqdm.tqdm(seq_groups.items()):
        # Sort by release date to find the first published structure
        info_group.sort(key=lambda x: x.get('release_date', 'z')) # 'z' ensures None is last
        rep_name = info_group[0]['name']
        for info in info_group:
            lookup[info['name']] = rep_name

    df['reference'] = df.index.map(lookup)

    # Save the master CSV
    print(f"Writing {len(df)} records to {args.outcsv}...")
    try:
        df.to_csv(args.outcsv, index=True, index_label='name')
        print("Master CSV file successfully written.")
    except Exception as e:
        print(f"An error occurred during CSV writing: {e}")

    return df

def _train_split(df):
    print("Creating train/val splits...")
    df['seqres'].fillna('', inplace=True)
    df['seqlen'] = [len(s) for s in df.seqres]
    df = df[df.valid_alphas > 0]
    df = df[df.saved]
    df = df[(df.seqlen >= 20) & (df.seqlen <= 256)]
    df = df[df.release_date < '2020-12-01']
    df['split'] = np.where(df.release_date < '2020-05-01', 'train', 'val')
    df.to_csv('splits/limit256.csv', index=True, index_label='name')
    print("Finished train/val splits.")

def _apo_split(df):
    print("Creating apo/holo splits...")
    df['seqres'].fillna('', inplace=True)
    df['seqlen'] = [len(s) for s in df.seqres]
    names = []
    apo = pd.read_csv('splits/revision1_86_plus_5.csv', sep=';')
    for _, row in apo.iterrows():
        name1 = row.apo_id[:4].lower() + '.' + row.apo_id[-1] + '.pdb'
        name2 = row.holo_id[:4].lower() + '.' + row.holo_id[-1] + '.pdb'
        if name1 not in df.index or name2 not in df.index: continue
        names.append(name1); names.append(name2)

    df = df.loc[names]
    names = []
    others = []
    for i in range(0, len(df), 2):
        if 'X' in df.seqres[i]: continue
        names.append(df.index[i]); others.append(df.index[i+1])

    df = df.loc[names]
    df['holo'] = others
    df_filtered = df.drop(columns=['reference', 'saved'], errors='ignore')
    df_filtered = df_filtered[df_filtered.seqlen <= 750]
    df_filtered.to_csv('splits/apo.csv', index=True, index_label='name')
    print("Finished apo/holo splits.")

def _codnas_split(df):
    print("Creating CoDNaS splits...")
    df['seqres'].fillna('', inplace=True)
    df['seqlen'] = [len(s) for s in df.seqres]
    names = []
    codnas = pd.read_csv('splits/codnas_orig.csv')
    for _, row in codnas.iterrows():
        name1 = row.Fold1[:4] + '.' + row.Fold1[-1] + '.pdb'
        name2 = row.Fold2[:4] + '.' + row.Fold2[-1] + '.pdb'
        if name1 not in df.index or name2 not in df.index: continue
        names.append(name1); names.append(name2)

    df = df.loc[names]
    names = []
    others = []
    for i in range(0, len(df), 2):
        maxlen = max(df.seqlen[i:i+2])
        minlen = min(df.seqlen[i:i+2])
        if maxlen / minlen > 1.5: continue
        if df.seqlen[i+1] < df.seqlen[i]:
            names.append(df.index[i+1]); others.append(df.index[i])
        else:
            names.append(df.index[i]); others.append(df.index[i+1])

    df = df.loc[names]
    df['other'] = others
    df_filtered = df.drop(columns=['reference', 'saved'], errors='ignore')
    df_filtered = df_filtered[df_filtered.seqlen <= 750]
    df_filtered.to_csv('splits/codnas.csv', index=True, index_label='name')
    print("Finished CoDNaS splits.")

def _cameo_split(df):
    print("Creating CAMEO splits...")
    df['seqres'].fillna('', inplace=True)
    df['seqlen'] = [len(s) for s in df.seqres]
    cameo = pd.read_csv('splits/cameo2022_orig.csv')
    cameo['name'] = [s.replace('[', '').replace(']', '').replace(' ', '.') + '.pdb' for s in cameo['ref. PDB [Chain]']]
    cameo = cameo.set_index('name').join(df[['release_date', 'seqres']], how='inner')

    tosave = cameo[(cameo.release_date < '2022-11-01') & (cameo.release_date >= '2022-08-01')]
    tosave = df.loc[tosave.index]
    df_filtered = tosave.drop(columns=['reference', 'saved'], errors='ignore')
    df_filtered = df_filtered[df_filtered.seqlen < 750]
    df_filtered.to_csv('splits/cameo2022.csv', index=True, index_label='name')
    print("Finished CAMEO splits.")

def create_final_splits(master_df):
    """
    Takes the master DataFrame and creates all the final data splits.
    """
    print("\nStage 3: Creating final data splits...")
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = master_df.copy()
    _train_split(df_copy)
    # _apo_split(df_copy)
    # _codnas_split(df_copy)
    # _cameo_split(df_copy)
    print("All data splits created successfully.")

def main():
    info_list = unpack_and_clean_pdbs()
    master_df = create_master_dataframe(info_list)
    if master_df is not None:
        create_final_splits(master_df)

if __name__ == "__main__":
    # Define argument parser here so it's available globally
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--manifest', type=str, default='./data/pdb.dat')
    arg_parser.add_argument('--data', type=str, default='./data/pdb')
    arg_parser.add_argument('--outdir', type=str, default='./data/pdb_chains')
    arg_parser.add_argument('--outcsv', type=str, default='./data/pdb_chains.csv')
    arg_parser.add_argument('--num_workers', type=int, default=15)
    args = arg_parser.parse_args()

    main()
