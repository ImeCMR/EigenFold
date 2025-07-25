import argparse
import numpy as np
from Bio.PDB import PDBParser, Selection
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def map_to_alpha_carbon(structure, res_num, chain_id):
    """Finds the alpha carbon of a given residue."""
    try:
        residue = structure[0][chain_id][res_num]
        return residue['CA'].get_coord()
    except KeyError:
        return None

def main():
    parser = argparse.ArgumentParser(description="Map NOESY contacts to backbone alpha carbons.")
    parser.add_argument("noesy_npz_file", help="Path to the NOESY contacts NPZ file.")
    parser.add_argument("pdb_file", help="Path to the corresponding PDB file (with hydrogens).")
    parser.add_argument("output_npz_file", help="Path to save the mapped backbone contacts NPZ file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load NOESY data
    try:
        noesy_data = np.load(args.noesy_npz_file)['noesy_data']
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Error loading NOESY data from {args.noesy_npz_file}: {e}")
        return

    # Load PDB structure
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(Path(args.pdb_file).stem, args.pdb_file)
    except FileNotFoundError:
        logger.error(f"PDB file not found: {args.pdb_file}")
        return

    mapped_contacts = []
    for contact in noesy_data:
        res1_num = contact['res1_num']
        res2_num = contact['res2_num']
        chain_id = contact['chain_id']

        # Only consider ILE, LEU, VAL residues
        res1 = structure[0][chain_id][res1_num].get_resname()
        res2 = structure[0][chain_id][res2_num].get_resname()
        if res1 not in ["ILE", "LEU", "VAL"] or res2 not in ["ILE", "LEU", "VAL"]:
            continue

        ca1_coord = map_to_alpha_carbon(structure, res1_num, chain_id)
        ca2_coord = map_to_alpha_carbon(structure, res2_num, chain_id)

        if ca1_coord is not None and ca2_coord is not None:
            distance = np.linalg.norm(ca1_coord - ca2_coord)
            mapped_contacts.append((res1_num, res2_num, distance, contact['peak_type']))

    if not mapped_contacts:
        logger.info("No mapped contacts to save.")
        return

    # Save mapped contacts
    mapped_dtype = np.dtype([
        ('res1_num', np.int32),
        ('res2_num', np.int32),
        ('distance', np.float32),
        ('peak_type', np.int8)
    ])
    mapped_contacts_np = np.array(mapped_contacts, dtype=mapped_dtype)

    try:
        np.savez_compressed(args.output_npz_file, backbone_contacts=mapped_contacts_np)
        logger.info(f"Saved {len(mapped_contacts_np)} mapped backbone contacts to {args.output_npz_file}")
    except Exception as e:
        logger.error(f"Failed to save mapped contacts NPZ file: {e}")

if __name__ == "__main__":
    main()
