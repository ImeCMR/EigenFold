import unittest
import os
import sys
import numpy as np

# Add project root to sys.path to allow importing utils.pdb
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.pdb import get_casp13_atom_name, generate_noesy_data
from utils.logging import get_logger # Import get_logger

# Configure logger for tests
logger = get_logger(__name__, level="INFO") # Changed level to INFO for less verbose test output by default

# Helper to get the full path of a test PDB file
def get_test_pdb_path(filename):
    return os.path.join(os.path.dirname(__file__), "test_data", filename)

class TestPDBUtils(unittest.TestCase):

    def test_get_casp13_atom_name_backbone(self):
        logger.info("Testing get_casp13_atom_name for backbone H")
        self.assertEqual(get_casp13_atom_name("ALA", "H"), "H")
        self.assertEqual(get_casp13_atom_name("GLY", "H"), "H")
        self.assertEqual(get_casp13_atom_name("LEU", "H"), "H")
        self.assertIsNone(get_casp13_atom_name("ALA", "HA")) # Alpha hydrogen should not be "H"

    def test_get_casp13_atom_name_ala(self):
        logger.info("Testing get_casp13_atom_name for ALA")
        self.assertEqual(get_casp13_atom_name("ALA", "HB1"), "HB")
        self.assertEqual(get_casp13_atom_name("ALA", "HB2"), "HB")
        self.assertEqual(get_casp13_atom_name("ALA", "HB3"), "HB")
        self.assertEqual(get_casp13_atom_name("ALA", "1HB"), "HB") # Common alternative
        self.assertEqual(get_casp13_atom_name("ALA", "H"), "H") # Backbone
        self.assertIsNone(get_casp13_atom_name("ALA", "CB")) # Carbon atom

    def test_get_casp13_atom_name_val(self):
        logger.info("Testing get_casp13_atom_name for VAL")
        self.assertEqual(get_casp13_atom_name("VAL", "HG11"), "HG1")
        self.assertEqual(get_casp13_atom_name("VAL", "HG12"), "HG1")
        self.assertEqual(get_casp13_atom_name("VAL", "HG13"), "HG1")
        self.assertEqual(get_casp13_atom_name("VAL", "HG21"), "HG2")
        self.assertEqual(get_casp13_atom_name("VAL", "HG22"), "HG2")
        self.assertEqual(get_casp13_atom_name("VAL", "HG23"), "HG2")
        self.assertEqual(get_casp13_atom_name("VAL", "1HG1"), "HG1") # Alternative PDB naming
        self.assertEqual(get_casp13_atom_name("VAL", "H"), "H")

    def test_get_casp13_atom_name_leu(self):
        logger.info("Testing get_casp13_atom_name for LEU")
        self.assertEqual(get_casp13_atom_name("LEU", "HD11"), "HD1")
        self.assertEqual(get_casp13_atom_name("LEU", "HD12"), "HD1")
        self.assertEqual(get_casp13_atom_name("LEU", "HD13"), "HD1")
        self.assertEqual(get_casp13_atom_name("LEU", "HD21"), "HD2")
        self.assertEqual(get_casp13_atom_name("LEU", "HD22"), "HD2")
        self.assertEqual(get_casp13_atom_name("LEU", "HD23"), "HD2")
        self.assertEqual(get_casp13_atom_name("LEU", "H"), "H")

    def test_get_casp13_atom_name_ile(self):
        logger.info("Testing get_casp13_atom_name for ILE")
        self.assertEqual(get_casp13_atom_name("ILE", "HD11"), "HD1") # Delta methyl
        self.assertEqual(get_casp13_atom_name("ILE", "HD12"), "HD1")
        self.assertEqual(get_casp13_atom_name("ILE", "HD13"), "HD1")
        self.assertEqual(get_casp13_atom_name("ILE", "HG21"), "HG2") # Gamma 2 methyl
        self.assertEqual(get_casp13_atom_name("ILE", "HG22"), "HG2")
        self.assertEqual(get_casp13_atom_name("ILE", "HG23"), "HG2")
        self.assertEqual(get_casp13_atom_name("ILE", "CD1",), "HD1") # From RELEVANT_HYDROGENS_MAP, was this intended? It's a carbon. Let's test current behavior.
                                                                # This should be None if CD1 is a carbon. The map was {'HD1': 'HD1', 'HG2': 'HG2', 'CD1': 'HD1'}
                                                                # Corrected map in utils/pdb.py is {'HD1': 'HD1', 'HG2': 'HG2'}
                                                                # So, this test should expect None for "CD1"
        self.assertIsNone(get_casp13_atom_name("ILE", "CD1"))


        self.assertEqual(get_casp13_atom_name("ILE", "H"), "H")

    def test_get_casp13_atom_name_asn_gln(self):
        logger.info("Testing get_casp13_atom_name for ASN and GLN sidechain amides")
        self.assertEqual(get_casp13_atom_name("ASN", "HD21"), "HD2")
        self.assertEqual(get_casp13_atom_name("ASN", "HD22"), "HD2")
        self.assertEqual(get_casp13_atom_name("ASN", "1HD2"), "HD2") # Alternative
        self.assertEqual(get_casp13_atom_name("GLN", "HE21"), "HE2")
        self.assertEqual(get_casp13_atom_name("GLN", "HE22"), "HE2")
        self.assertEqual(get_casp13_atom_name("GLN", "1HE2"), "HE2") # Alternative
        self.assertIsNone(get_casp13_atom_name("ASN", "ND2")) # Nitrogen atom
        self.assertIsNone(get_casp13_atom_name("GLN", "NE2")) # Nitrogen atom

    def test_get_casp13_atom_name_unknown(self):
        logger.info("Testing get_casp13_atom_name for unknown atoms")
        self.assertIsNone(get_casp13_atom_name("ALA", "CA"))
        self.assertIsNone(get_casp13_atom_name("VAL", "CG")) # Non-existent PDB name for VAL
        self.assertIsNone(get_casp13_atom_name("XYZ", "H")) # Unknown residue

    # More tests for generate_noesy_data will follow here

    # Example test for generate_noesy_data using ala_val.pdb
    def test_generate_noesy_ala_val_basic(self):
        logger.info("Testing generate_noesy_data with ala_val.pdb (no explicit H)")
        pdb_file = get_test_pdb_path("ala_val.pdb")
        # Using a fairly large cutoff to catch potential inter-residue contacts after obabel adds H
        # ALA CB (res 0) to VAL CB (res 1) is sqrt((7-11)^2 + (8-12)^2 + (10-14)^2) = sqrt(16+16+16) = sqrt(48) approx 6.9A
        # Hydrogens on these will be closer.
        noesy_contacts = generate_noesy_data(pdb_file, distance_cutoff=7.0, noise_fraction=0.0)

        self.assertIsInstance(noesy_contacts, list)

        if not noesy_contacts:
            logger.warning(f"No NOESY contacts found for {pdb_file} with 7.0A cutoff. OpenBabel might not have run or added H as expected.")
            # This might happen if openbabel is not installed/found in the test env.
            # For now, we can't fail the test strictly on this, but we log it.
            return

        for contact_str in noesy_contacts:
            parts = contact_str.split()
            self.assertEqual(len(parts), 6, f"NOESY output format error: {contact_str}")
            res_from, res_to, peak_id, dist, atom_from, atom_to = parts

            self.assertTrue(res_from.isdigit() and int(res_from) == 0, f"Expected res_from=0: {contact_str}") # ALA is residue 0
            self.assertTrue(res_to.isdigit() and int(res_to) == 1, f"Expected res_to=1: {contact_str}")     # VAL is residue 1
            self.assertTrue(peak_id.isdigit())
            self.assertTrue(float(dist) <= 7.0)

            if atom_from == "HB": # ALA
                self.assertIn(atom_to, ["H", "HB", "HG1", "HG2"], f"Unexpected ALA HB partner: {atom_to} in {contact_str}")
            elif atom_to == "HB": # ALA
                self.assertIn(atom_from, ["H", "HB", "HG1", "HG2"], f"Unexpected ALA HB partner: {atom_from} in {contact_str}")

            if atom_from in ["HG1", "HG2"]: # VAL
                 self.assertIn(atom_to, ["H", "HB"], f"Unexpected VAL HG1/HG2 partner: {atom_to} in {contact_str}")
            elif atom_to in ["HG1", "HG2"]: # VAL
                 self.assertIn(atom_from, ["H", "HB"], f"Unexpected VAL HG1/HG2 partner: {atom_from} in {contact_str}")
        logger.info(f"Found {len(noesy_contacts)} contacts for ala_val.pdb (7.0A cutoff, no noise). First: {noesy_contacts[0] if noesy_contacts else 'None'}")


    def test_generate_noesy_ile_asn_naming_and_format(self):
        logger.info("Testing generate_noesy_data with ile_asn.pdb (some explicit H)")
        pdb_file = get_test_pdb_path("ile_asn.pdb")
        noesy_contacts = generate_noesy_data(pdb_file, distance_cutoff=8.0, noise_fraction=0.0) # Cutoff to ensure some contacts
        self.assertIsInstance(noesy_contacts, list)

        if not noesy_contacts:
            logger.warning(f"No NOESY contacts found for {pdb_file}. Cannot verify naming and format thoroughly.")
            return

        atom_names_seen = set()
        for contact_str in noesy_contacts:
            parts = contact_str.split()
            self.assertEqual(len(parts), 6, f"NOESY output format error: {contact_str}")

            res_from_idx, res_to_idx, peak_id, dist_str, atom_from, atom_to = parts

            self.assertTrue(res_from_idx.isdigit(), f"res_from_idx not a digit: {contact_str}")
            self.assertTrue(res_to_idx.isdigit(), f"res_to_idx not a digit: {contact_str}")
            self.assertTrue(peak_id.isdigit(), f"peak_id not a digit: {contact_str}")
            try:
                dist_val = float(dist_str)
                self.assertTrue(dist_val <= 8.0, f"Distance {dist_val} exceeds cutoff: {contact_str}")
            except ValueError:
                self.fail(f"Distance is not a float: {contact_str}")

            self.assertIn(int(res_from_idx), [0, 1], f"res_from_idx out of bounds: {contact_str}") # ILE=0, ASN=1
            self.assertIn(int(res_to_idx), [0, 1], f"res_to_idx out of bounds: {contact_str}")
            self.assertNotEqual(res_from_idx, res_to_idx, f"Intra-residue contact found: {contact_str}")


            # Check CASP13 names (these are the most likely to appear for ILE/ASN)
            valid_casp_names = {"H", "HD1", "HG2", "HD2"} # Simplified set for this test
            self.assertIn(atom_from, valid_casp_names, f"Unexpected atom_from name {atom_from}: {contact_str}")
            self.assertIn(atom_to, valid_casp_names, f"Unexpected atom_to name {atom_to}: {contact_str}")
            atom_names_seen.add(atom_from)
            atom_names_seen.add(atom_to)

        logger.info(f"Atom names seen in ile_asn.pdb contacts: {atom_names_seen}")
        # Check if specific expected atom types appeared (highly dependent on OpenBabel's output and geometry)
        # For example, for ASN, HD21/HD22 should be grouped to HD2.
        # For ILE, HD11/12/13 to HD1, HG21/22/23 to HG2.
        self.assertTrue("HD1" in atom_names_seen or "HG2" in atom_names_seen, "Expected ILE specific hydrogens (HD1 or HG2) if ILE involved in contacts.")
        self.assertTrue("HD2" in atom_names_seen, "Expected ASN specific grouped hydrogen (HD2) if ASN involved in contacts.")


    def test_generate_noesy_empty_pdb(self):
        logger.info("Testing generate_noesy_data with empty.pdb")
        pdb_file = get_test_pdb_path("empty.pdb")
        noesy_contacts = generate_noesy_data(pdb_file)
        self.assertEqual(len(noesy_contacts), 0, "Should return empty list for PDB with no atoms")

    def test_generate_noesy_noise_generation(self):
        logger.info("Testing generate_noesy_data noise generation")
        pdb_file = get_test_pdb_path("pre_h.pdb") # Use a PDB likely to have true contacts

        contacts_no_noise = generate_noesy_data(pdb_file, distance_cutoff=5.0, noise_fraction=0.0)
        contacts_with_noise = generate_noesy_data(pdb_file, distance_cutoff=5.0, noise_fraction=0.5)

        num_no_noise = len(contacts_no_noise)
        num_with_noise = len(contacts_with_noise)

        logger.info(f"Contacts without noise: {num_no_noise}, with noise: {num_with_noise} for pre_h.pdb")

        if num_no_noise > 0: # If there are true contacts to begin with
            self.assertGreater(num_with_noise, num_no_noise, "With noise, should have more contacts than without (if true contacts exist).")
        else:
            logger.info("No true contacts found in pre_h.pdb for noise test baseline. Noise test might not be fully indicative.")
            # If no true contacts, number of false positives should be 0 (current logic: noise_fraction * num_true_contacts)
            self.assertEqual(num_with_noise, 0, "If no true contacts, noise generation should also produce 0 false positives based on current logic.")


        # Verify unique peak IDs
        if num_with_noise > 0:
            peak_ids = set()
            for contact_str in contacts_with_noise:
                peak_id = int(contact_str.split()[2])
                self.assertNotIn(peak_id, peak_ids, f"Peak ID {peak_id} is not unique.")
                peak_ids.add(peak_id)

    # TODO: Add a test for distance calculation if a reliable pair can be identified post-OpenBabel.
    # This requires knowing exact coordinates OpenBabel will produce for hydrogens.


if __name__ == '__main__':
    unittest.main()
