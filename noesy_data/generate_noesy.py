import argparse
import os
import random
import subprocess
import tempfile
import logging
import traceback
from pathlib import Path
import shutil
import itertools # Added for combinations

from Bio.PDB import PDBParser, Structure as BioPDBStructure # Explicit import for Structure
from Bio.PDB.vectors import Vector
import numpy as np

logger = logging.getLogger(__name__)

# --- Global Constants ---
ATOMIC_NUMBER_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S'}

# RELEVANT_ATOMS is for the old get_atoms, might be unused now by extract_filtered_protons logic directly
# but could be kept if some parts of the script (not NOESY generation) might use it.
# For now, keep it as it was in the user's last full script.
RELEVANT_ATOMS = {
    'ALA': ['HB1', 'HB2', 'HB3'],
    'ARG': ['HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE', 'NH1', 'NH2'],
    'ASN': ['HB2', 'HB3', 'HD21', 'HD22'],
    'ASP': ['HB2', 'HB3'],
    'CYS': ['HB2', 'HB3', 'HG'],
    'GLN': ['HB2', 'HB3', 'HG2', 'HG3', 'HE21', 'HE22'],
    'GLU': ['HB2', 'HB3', 'HG2', 'HG3'],
    'GLY': ['HA2', 'HA3'],
    'HIS': ['HB2', 'HB3', 'HD2', 'HE1'],
    'ILE': ['HG21', 'HG22', 'HG23', 'HD11', 'HD12', 'HD13', 'HG12', 'HG13'],
    'LEU': ['HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23'],
    'LYS': ['HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 'HZ3'],
    'MET': ['HB2', 'HB3', 'HG2', 'HG3', 'HE1', 'HE2', 'HE3'],
    'PHE': ['HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ'],
    'PRO': ['HB2', 'HB3', 'HG2', 'HG3', 'HD2', 'HD3'],
    'SER': ['HB2', 'HB3', 'HG'],
    'THR': ['HB', 'HG21', 'HG22', 'HG23', 'HG1'],
    'TRP': ['HB2', 'HB3', 'HD1', 'HE1', 'HZ2', 'HZ3', 'HH2', 'HE3'],
    'TYR': ['HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HH'],
    'VAL': ['HB', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23'],
}
BACKBONE_AMIDE = "H" # Potentially unused by new proton extraction

PDB2PQR_PATH = "/orange/alberto.perezant/imesh.ranaweera/noesy_project/boltzNOESY/boltz/boltznoesy_env/bin/pdb2pqr30"

# Parameters for new NOESY methodology
H_MATCH_TOLERANCE = 0.03  # ppm
NEW_DISTANCE_CUTOFF = 7.5     # Default for initial pair consideration (Angstroms)
DISTANCE_NOE_THRESHOLD = 5.0  # For classifying as NOE type (1) vs. Ambiguous type (0)
NOISE_STD_H_SHIFT_SIM = 0.01  # ppm, for shift simulation noise

TARGET_HYDROPHOBIC_RESIDUES = {"ALA", "ILE", "LEU", "MET", "PHE", "PRO", "TRP", "VAL"}

# --- Helper Functions ---

def add_hydrogens(pdb_file: str, output_pdb_file: str) -> bool:
    # This function remains as previously consolidated (correct PDB2PQR_PATH, args, cleanup)
    print(f"DEBUG: Entering add_hydrogens for {pdb_file}, output: {output_pdb_file}", flush=True)
    dummy_pqr_output_path = output_pdb_file + ".pqr_dummy"
    command = [ PDB2PQR_PATH, "--ff=AMBER", "--pdb-output", output_pdb_file, pdb_file, dummy_pqr_output_path ]
    logger.info(f"Calling pdb2pqr30 for {pdb_file}...")
    logger.info(f"Command: {' '.join(command)}")
    status = False
    try:
        completed_process = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
        if completed_process.returncode != 0:
            logger.error(f"pdb2pqr30 failed for {pdb_file} code {completed_process.returncode}")
            logger.error(f"stdout:\n{completed_process.stdout}")
            logger.error(f"stderr:\n{completed_process.stderr}")
            status = False
        else:
            logger.info(f"pdb2pqr30 completed successfully for {pdb_file}.")
            if not os.path.exists(output_pdb_file) or os.path.getsize(output_pdb_file) == 0:
                logger.error(f"Output file {output_pdb_file} missing or empty post pdb2pqr30.")
                status = False
            else:
                status = True
    except subprocess.TimeoutExpired as e:
        logger.error(f"pdb2pqr30 timed out for {pdb_file}: {e}")
        status = False
    except FileNotFoundError:
        logger.error(f"pdb2pqr30 not found at {PDB2PQR_PATH}.")
        status = False
    except Exception as e:
        logger.error(f"pdb2pqr30 error for {pdb_file}: {e}", exc_info=True)
        status = False
    finally:
        if os.path.exists(dummy_pqr_output_path):
            try:
                os.remove(dummy_pqr_output_path)
                logger.debug(f"Cleaned dummy PQR: {dummy_pqr_output_path}")
            except OSError as e_remove:
                logger.warning(f"Could not remove dummy PQR {dummy_pqr_output_path}: {e_remove}")
    return status

def decode_atom_name_from_4i1(encoded_name: np.ndarray) -> str:
    # This function remains as previously consolidated (refined padding)
    try:
        chars = []
        for val in encoded_name:
            if val == 0: break
            char = chr(int(val) + 32)
            chars.append(char)
        name = "".join(chars).strip()
    except Exception as e:
        logger.error(f"Error decoding atom name from {encoded_name!r}: {e}. Using 'ERR '.")
        return "ERR "
    if not name:
        logger.warning(f"Decoded atom name from {encoded_name!r} empty. Using 'UNK '.")
        return "UNK "
    name_len = len(name)
    if name_len == 1: return f" {name}  "
    elif name_len == 2:
        if name[0].isalpha(): return f" {name} "
        else: return f"{name:<4}"
    elif name_len == 3:
        if name == "OXT": return " OXT"
        else: return f"{name:<4}"
    elif name_len == 4: return name
    else:
        logger.warning(f"Decoded name '{name}' from {encoded_name!r} >4 chars. Truncating.")
        return name[:4]

def parse_pdb(pdb_file: str) -> BioPDBStructure:
    """Parses a PDB file and returns a BioPython Structure object."""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(Path(pdb_file).stem, pdb_file)
        return structure
    except Exception as e:
        logger.error(f"Error parsing PDB file {pdb_file}: {e}")
        raise

def write_temp_pdb_from_npz(npz_data: dict, temp_pdb_path: str):
    # This function remains as previously consolidated (protein-only, 000-coord filter, PDB formatting)
    print(f"DEBUG: write_temp_pdb_from_npz for {temp_pdb_path}", flush=True)
    atoms_data, residues_data, chains_data = npz_data['atoms'], npz_data['residues'], npz_data['chains']
    atom_serial = atoms_written_count = 0
    with open(temp_pdb_path, 'w', encoding='utf-8') as f:
        for ci, chain_entry in enumerate(chains_data):
            chain_processed_atom_count = 0
            if len(chain_entry) < 9: logger.warning(f"Chain {ci} too short. Skip."); continue
            chain_pdb_id = (str(chain_entry[0]).strip() or 'A')[0]
            res_start_idx, num_res_chain = int(chain_entry[7]), int(chain_entry[8])
            for res_offset in range(num_res_chain):
                res_global_idx = res_start_idx + res_offset
                if res_global_idx >= len(residues_data): logger.warning(f"Res idx {res_global_idx} out of bounds. Skip."); continue
                res_entry = residues_data[res_global_idx]
                if len(res_entry) < 8: logger.warning(f"Res entry {res_global_idx} too short. Skip."); continue
                res_name, res_seq_pdb = str(res_entry[0]), int(res_entry[2])+1
                atom_start_global, num_atoms_res = int(res_entry[3]), int(res_entry[4])
                is_standard = bool(res_entry[7])
                if not is_standard: continue # Protein-only filter
                record_type = "ATOM  "
                for atom_offset_res in range(num_atoms_res):
                    atom_global_idx = atom_start_global + atom_offset_res
                    if atom_global_idx >= len(atoms_data): logger.warning(f"Atom idx {atom_global_idx} out of bounds. Skip."); continue
                    atom_npz_entry = atoms_data[atom_global_idx]
                    if len(atom_npz_entry) < 4: logger.warning(f"Atom entry {atom_global_idx} too short. Skip."); continue
                    coords_list = atom_npz_entry[3]
                    if not hasattr(coords_list, '__getitem__') or len(coords_list) < 3: logger.warning(f"Atom {atom_global_idx} bad coords. Skip."); continue
                    x,y,z = coords_list[0],coords_list[1],coords_list[2]
                    if x==0.0 and y==0.0 and z==0.0: logger.info(f"Atom (pot. serial {atom_serial+1}) Res:{res_name}{res_seq_pdb} NPZ idx:{atom_global_idx} (0,0,0) coords. Skip."); continue
                    encoded_name = atom_npz_entry[0]
                    if not hasattr(encoded_name, '__iter__') or not encoded_name.any():
                        logger.warning(f"Atom {atom_global_idx} invalid encoded name: {encoded_name!r}. Using 'UNK '.")
                        atom_name_pdb = "UNK "
                    else: atom_name_pdb = decode_atom_name_from_4i1(encoded_name)
                    atomic_num = atom_npz_entry[1]
                    element = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_num, 'X').rjust(2)
                    atom_serial+=1; chain_processed_atom_count+=1; atoms_written_count+=1
                    alt_loc, icode = ' ', ' '
                    pdb_line = (f"{record_type:<6s}{atom_serial:5d} {atom_name_pdb}{alt_loc}"
                                f"{res_name[:3]:<3s} {chain_pdb_id:1s}{res_seq_pdb:4d}{icode}   "
                                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          "
                                f"{element:>2s}  \n")
                    f.write(pdb_line)
                    if atoms_written_count % 1000 == 0 and atoms_written_count > 0: logger.info(f"DEBUG: ... {atoms_written_count} atoms written ...")
            if chain_processed_atom_count > 0: f.write(f"TER   {atom_serial+1:5d}      {res_name[:3]:<3s} {chain_pdb_id:1s}{res_seq_pdb:4d}\n")
        f.write("END\n")
    print(f"DEBUG: Exiting write_temp_pdb_from_npz, total atoms written: {atoms_written_count}", flush=True)
    if atoms_written_count == 0: logger.warning(f"No atoms written to PDB: {temp_pdb_path}.")

# Old get_atoms is commented out
# def get_atoms(structure): ...

def extract_filtered_protons(structure: BioPDBStructure) -> list[dict]:
    # This function remains as previously consolidated
    protons = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()
                if res_name not in TARGET_HYDROPHOBIC_RESIDUES: continue
                if residue.get_id()[0] != ' ': continue
                try: residue_number = int(residue.get_id()[1])
                except ValueError: logger.warning(f"Cannot parse res num for {res_name}{residue.get_id()} C:{chain.id}. Skip."); continue
                for atom in residue:
                    is_proton = False; atom_name_stripped = atom.get_id().strip().upper()
                    if atom.element == 'H': is_proton = True
                    elif atom_name_stripped.startswith('H'): is_proton = True
                    if is_proton:
                        protons.append({'atom_obj':atom,'coord':atom.coord,'res_num':residue_number,'chain_id':chain.id,'atom_name':atom.get_id().strip()})
    return protons

def simulate_shift(coord: np.ndarray) -> float:
    # This function remains as previously consolidated
    base_shift = np.linalg.norm(coord) * 0.1
    noise = np.random.normal(loc=0.0, scale=NOISE_STD_H_SHIFT_SIM)
    return round(base_shift + noise, 3)

# Old generate_noesy_data is commented out
# def generate_noesy_data(pdb_file_with_hydrogens, distance_cutoff=5.0): ...

def compute_contacts_new_method(protons: list[dict], initial_distance_cutoff: float, actual_noe_distance_threshold: float) -> list[dict]:
    # Signature updated, uses passed cutoffs
    contacts = []
    if not protons or len(protons) < 2:
        logger.info("Not enough protons for contact computation.")
        return contacts
    for p1, p2 in itertools.combinations(protons, 2):
        if p1['chain_id'] != p2['chain_id']: continue
        if p1['res_num'] == p2['res_num']: continue # Intra-residue check
        distance = np.linalg.norm(p1['coord'] - p2['coord'])
        if distance > initial_distance_cutoff: continue # Use passed parameter
        shift1 = simulate_shift(p1['coord'])
        shift2 = simulate_shift(p2['coord'])
        if abs(shift1 - shift2) > H_MATCH_TOLERANCE: continue
        peak_type = 1 if distance <= actual_noe_distance_threshold else 0 # Use passed parameter
        contacts.append({
            'chain_id': p1['chain_id'], 'res1_num': p1['res_num'], 'atom1_name': p1['atom_name'],
            'res2_num': p2['res_num'], 'atom2_name': p2['atom_name'],
            'distance': round(distance, 2), 'peak_type': peak_type
        })
    logger.info(f"Computed {len(contacts)} contacts via new method.")
    return contacts

def main():
    parser = argparse.ArgumentParser(description="Generate NOESY-like contacts from PDB files using new methodology.")
    parser.add_argument("input_dir", help="Directory containing input PDB files.")
    parser.add_argument("output_dir", help="Directory to save generated contact data.")
    parser.add_argument("--distance_cutoff", type=float, default=7.5,
                        help="Initial distance cutoff for proton pair consideration (Angstroms). Default: 7.5. True NOE classification uses an internal threshold of 5.0 A.")
    parser.add_argument("--pdb2pqr-path", help="Path to the pdb2pqr executable.", default="/orange/alberto.perezant/imesh.ranaweera/noesy_project/boltzNOESY/boltz/boltznoesy_env/bin/pdb2pqr30")
    args = parser.parse_args()

    global PDB2PQR_PATH
    PDB2PQR_PATH = args.pdb2pqr_path

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir); logger.info(f"Created output dir: {args.output_dir}")
    pdb_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".pdb")]
    if not pdb_files: logger.warning(f"No .pdb files in {args.input_dir}"); return

    pdb_parser = PDBParser(QUIET=True)

    noesy_contact_dtype = np.dtype([
        ('chain_id', 'U1'),       # Single character for chain
        ('res1_num', np.int32),   # Residue number 1
        ('res2_num', np.int32),   # Residue number 2
        ('peak_type', np.int8),   # 0 or 1
        ('distance', np.float32), # Distance
        ('atom1_name', 'U4'),     # Atom name (max 4 chars, e.g. "HD11")
        ('atom2_name', 'U4')      # Atom name (max 4 chars)
    ])

    for pdb_filename in pdb_files:
        pdb_file_path = os.path.join(args.input_dir, pdb_filename)
        name_part_pdb, _ = os.path.splitext(pdb_filename)
        logger.info(f"\nProcessing {pdb_file_path}...")
        temp_hydro_pdb_name = None
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='_hydro.pdb', encoding='utf-8') as t_hydro:
                temp_hydro_pdb_name = t_hydro.name

            if not add_hydrogens(pdb_file_path, temp_hydro_pdb_name):
                logger.error(f"H addition failed for {pdb_file_path}. Skipping."); continue
            logger.info(f"Successfully added hydrogens: {temp_hydro_pdb_name}")

            structure = None
            try:
                structure = pdb_parser.get_structure(f"hydro_{name_part_pdb}", temp_hydro_pdb_name)
            except Exception as e_parse:
                logger.error(f"Failed to parse H-PDB {temp_hydro_pdb_name}: {e_parse}", exc_info=True); continue
            if not structure: logger.error(f"Parsed H-PDB {temp_hydro_pdb_name} is None. Skip."); continue

            protons = extract_filtered_protons(structure)
            logger.info(f"Extracted {len(protons)} protons for {name_part_pdb}.")

            contacts = compute_contacts_new_method(protons,
                                                   initial_distance_cutoff=args.distance_cutoff,
                                                   actual_noe_distance_threshold=DISTANCE_NOE_THRESHOLD)

            # Introduce false peaks
            num_false_peaks = int(len(contacts) * 0.3)
            for _ in range(num_false_peaks):
                p1, p2 = random.sample(protons, 2)
                distance = np.linalg.norm(p1['coord'] - p2['coord'])
                contacts.append({
                    'chain_id': p1['chain_id'], 'res1_num': p1['res_num'], 'atom1_name': p1['atom_name'],
                    'res2_num': p2['res_num'], 'atom2_name': p2['atom_name'],
                    'distance': round(distance, 2), 'peak_type': 0
                })

            if contacts:
                output_filename = os.path.join(args.output_dir, f"{name_part_pdb}.npz")
                contact_data_tuples = [] # Changed variable name for clarity
                for contact_dict in contacts:
                    # Ensure atom names are correctly sized for U4 dtype
                    atom1 = str(contact_dict['atom1_name'])[:4]
                    atom2 = str(contact_dict['atom2_name'])[:4]
                    chain_id_str = str(contact_dict['chain_id'])[:1] # Ensure single char for U1

                    contact_data_tuples.append((
                        chain_id_str,
                        contact_dict['res1_num'],
                        contact_dict['res2_num'],
                        contact_dict['peak_type'],
                        contact_dict['distance'], # np.array will cast to float32
                        atom1,
                        atom2
                    ))

                np_contact_data = np.array(contact_data_tuples, dtype=noesy_contact_dtype) # Use the new structured dtype

                try:
                    np.savez_compressed(output_filename, noesy_data=np_contact_data)
                    logger.info(f"Saved {len(contacts)} contacts for {name_part_pdb} to NPZ file: {output_filename}")
                except Exception as e_save:
                    logger.error(f"Failed to save NPZ file {output_filename}: {e_save}", exc_info=True)

            else:
                logger.info(f"No contacts generated for {name_part_pdb}, so no NPZ file will be saved.")
        except Exception as e:
            logger.error(f"Error processing file {pdb_file_path}: {e}", exc_info=True)
        finally:
            if temp_hydro_pdb_name and os.path.exists(temp_hydro_pdb_name):
                try: os.remove(temp_hydro_pdb_name)
                except OSError as e_os: logger.error(f"Error removing {temp_hydro_pdb_name}: {e_os}")
    logger.info("\nProcessing complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
