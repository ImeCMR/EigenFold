import argparse
import os
import random
import subprocess
import tempfile
import logging
import traceback
from pathlib import Path
import shutil
import itertools

from Bio.PDB import PDBParser, Structure as BioPDBStructure
from Bio.PDB.vectors import Vector
import numpy as np

logger = logging.getLogger(__name__)

# --- Global Constants ---
ATOMIC_NUMBER_TO_SYMBOL = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S'}

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
BACKBONE_AMIDE = "H"

PDB2PQR_PATH = "pdb2pqr30"  # Assuming pdb2pqr30 is in the system's PATH

# Parameters for new NOESY methodology
H_MATCH_TOLERANCE = 0.03  # ppm
NEW_DISTANCE_CUTOFF = 7.5  # Default for initial pair consideration (Angstroms)
DISTANCE_NOE_THRESHOLD = 6.0  # For classifying as NOE type (1) vs. Ambiguous type (0)
NOISE_STD_H_SHIFT_SIM = 0.01  # ppm, for shift simulation noise
FALSE_PEAK_RATIO = 0.3

TARGET_HYDROPHOBIC_RESIDUES = {"ILE", "LEU", "VAL"}


# --- Helper Functions ---
def add_hydrogens(pdb_file: str, output_pdb_file: str) -> bool:
    print(f"DEBUG: Entering add_hydrogens for {pdb_file}, output: {output_pdb_file}", flush=True)
    dummy_pqr_output_path = output_pdb_file + ".pqr_dummy"
    command = [
        PDB2PQR_PATH,
        "--ff=AMBER",
        "--pdb-output",
        output_pdb_file,
        pdb_file,
        dummy_pqr_output_path
    ]
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


def extract_filtered_protons(structure: BioPDBStructure) -> list[dict]:
    protons = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()
                if res_name not in TARGET_HYDROPHOBIC_RESIDUES:
                    continue
                if residue.get_id()[0] != ' ':
                    continue
                try:
                    residue_number = int(residue.get_id()[1])
                except ValueError:
                    logger.warning(f"Cannot parse res num for {res_name}{residue.get_id()} C:{chain.id}. Skip.");
                    continue
                for atom in residue:
                    is_proton = False
                    atom_name_stripped = atom.get_id().strip().upper()
                    if atom.element == 'H':
                        is_proton = True
                    elif atom_name_stripped.startswith('H'):
                        is_proton = True
                    if is_proton:
                        protons.append({
                            'atom_obj': atom,
                            'coord': atom.coord,
                            'res_num': residue_number,
                            'chain_id': chain.id,
                            'atom_name': atom.get_id().strip(),
                            'res_name': res_name
                        })
    return protons


def compute_contacts(protons: list[dict], distance_cutoff: float) -> list[dict]:
    contacts = []
    if not protons or len(protons) < 2:
        logger.info("Not enough protons for contact computation.")
        return contacts
    for p1, p2 in itertools.combinations(protons, 2):
        if p1['chain_id'] != p2['chain_id']:
            continue
        if p1['res_num'] == p2['res_num']:
            continue  # Intra-residue check
        distance = np.linalg.norm(p1['coord'] - p2['coord'])
        if distance > distance_cutoff:
            continue
        contacts.append({
            'res1_num': p1['res_num'],
            'atom1_name': p1['atom_name'],
            'res2_num': p2['res_num'],
            'atom2_name': p2['atom_name'],
            'distance': round(distance, 2),
        })
    logger.info(f"Computed {len(contacts)} contacts.")
    return contacts


def add_false_peaks(contacts: list[dict], protons: list[dict], false_peak_ratio: float) -> list[dict]:
    num_false_peaks = int(len(contacts) * false_peak_ratio)
    if num_false_peaks == 0:
        return contacts

    false_contacts = []
    for _ in range(num_false_peaks):
        p1, p2 = random.sample(protons, 2)
        if p1['chain_id'] != p2['chain_id'] or p1['res_num'] == p2['res_num']:
            continue

        distance = np.linalg.norm(p1['coord'] - p2['coord'])
        # Add a small random value to the distance to make it look more realistic
        distance += random.uniform(-0.5, 0.5)
        false_contacts.append({
            'res1_num': p1['res_num'],
            'atom1_name': p1['atom_name'],
            'res2_num': p2['res_num'],
            'atom2_name': p2['atom_name'],
            'distance': round(distance, 2),
        })
    logger.info(f"Added {len(false_contacts)} false peaks.")
    return contacts + false_contacts


def main():
    parser = argparse.ArgumentParser(description="Generate NOESY-like contacts from PDB files.")
    parser.add_argument("input_dir", help="Directory containing input PDB files.")
    parser.add_argument("output_dir", help="Directory to save generated contact data.")
    parser.add_argument("--distance_cutoff", type=float, default=6.0,
                        help="Distance cutoff for proton pair consideration (Angstroms).")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"Created output dir: {args.output_dir}")

    pdb_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".pdb")]
    if not pdb_files:
        logger.warning(f"No .pdb files in {args.input_dir}");
        return

    pdb_parser = PDBParser(QUIET=True)

    noesy_contact_dtype = np.dtype([
        ('res1_num', np.int32),
        ('res2_num', np.int32),
        ('distance', np.float32),
        ('atom1_name', 'U4'),
        ('atom2_name', 'U4')
    ])

    for pdb_filename in pdb_files:
        pdb_file_path = os.path.join(args.input_dir, pdb_filename)
        name_part_pdb, _ = os.path.splitext(pdb_filename)
        logger.info(f"\nProcessing {pdb_file_path}...")
        temp_hydro_pdb_name = None
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='_hydro.pdb',
                                             encoding='utf-8') as t_hydro:
                temp_hydro_pdb_name = t_hydro.name

            if not add_hydrogens(pdb_file_path, temp_hydro_pdb_name):
                logger.error(f"H addition failed for {pdb_file_path}. Skipping.");
                continue
            logger.info(f"Successfully added hydrogens: {temp_hydro_pdb_name}")

            structure = None
            try:
                structure = pdb_parser.get_structure(f"hydro_{name_part_pdb}", temp_hydro_pdb_name)
            except Exception as e_parse:
                logger.error(f"Failed to parse H-PDB {temp_hydro_pdb_name}: {e_parse}", exc_info=True);
                continue
            if not structure:
                logger.error(f"Parsed H-PDB {temp_hydro_pdb_name} is None. Skip.");
                continue

            protons = extract_filtered_protons(structure)
            logger.info(f"Extracted {len(protons)} protons for {name_part_pdb}.")

            contacts = compute_contacts(protons, args.distance_cutoff)
            contacts = add_false_peaks(contacts, protons, FALSE_PEAK_RATIO)

            if contacts:
                output_filename = os.path.join(args.output_dir, f"{name_part_pdb}.npz")
                contact_data_tuples = []
                for contact_dict in contacts:
                    atom1 = str(contact_dict['atom1_name'])[:4]
                    atom2 = str(contact_dict['atom2_name'])[:4]
                    contact_data_tuples.append((
                        contact_dict['res1_num'],
                        contact_dict['res2_num'],
                        contact_dict['distance'],
                        atom1,
                        atom2
                    ))

                np_contact_data = np.array(contact_data_tuples, dtype=noesy_contact_dtype)

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
                try:
                    os.remove(temp_hydro_pdb_name)
                except OSError as e_os:
                    logger.error(f"Error removing {temp_hydro_pdb_name}: {e_os}")
    logger.info("\nProcessing complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
