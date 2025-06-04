import torch, os, warnings, io, subprocess
from Bio.PDB import PDBIO, Chain, Residue, Polypeptide, Atom, PDBParser, PDBExceptions
from Bio import pairwise2
import numpy as np
# PDBConstructionWarning is already imported via PDBExceptions
# from Bio.PDB.PDBExceptions import PDBConstructionWarning
parser = PDBParser()
from .logging import get_logger
logger = get_logger(__name__)

# CASP13 atom naming convention
# HB for ALA methyl
# HG1, HG2 for VAL methyls
# HD1, HD2 for LEU methyls
# HD1 for ILE delta-methyl (and HG2 for the other ILE methyl)
# QG/QD for Val/Leu when methyls have same resonance (Placeholder - requires NMR data)
# Backbone amide HN (typically 'H' bonded to 'N') should be named 'H'.
# Asn/Gln sidechain amide hydrogens (HD21/HD22 for ASN, HE21/HE22 for GLN) should be grouped and named HD2 (ASN) or HE2 (GLN).

RELEVANT_HYDROGENS_MAP = {
    'ALA': {'HB': 'HB'},  # HB1, HB2, HB3 -> HB
    'VAL': {'HG1': 'HG1', 'HG2': 'HG2'}, # HG11, HG12, HG13 -> HG1; HG21, HG22, HG23 -> HG2
    'LEU': {'HD1': 'HD1', 'HD2': 'HD2'}, # HD11, HD12, HD13 -> HD1; HD21, HD22, HD23 -> HD2
    'ILE': {'HD1': 'HD1', 'HG2': 'HG2'}, # HD11, HD12, HD13 -> HD1; HG21, HG22, HG23 -> HG2
    # ASN HD21, HD22 are grouped to HD2
    # GLN HE21, HE22 are grouped to HE2
}
BACKBONE_AMIDE_ATOM_NAME = 'H' # Standard PDB name for backbone amide hydrogen
CASP13_BACKBONE_AMIDE_NAME = 'H'

# Placeholder for QG/QD naming for VAL/LEU if resonance data becomes available.
# This function would need to be significantly more complex, likely taking resonance assignments as input.
# For now, we will name individual methyls.

def get_casp13_atom_name(residue_name, atom_name):
    """
    Converts a PDB atom name to its CASP13 equivalent based on the residue and atom type.
    Handles specific naming conventions for methyl groups and amide hydrogens.
    """
    res_name_upper = residue_name.upper()
    atom_name_upper = atom_name.upper()

    # Backbone amide hydrogen
    if atom_name_upper == BACKBONE_AMIDE_ATOM_NAME:
        # Add check if it's bonded to 'N' if possible from PDB structure object,
        # for now, assume 'H' is the backbone amide.
        return CASP13_BACKBONE_AMIDE_NAME

    # ASN/GLN sidechain amide hydrogens
    if res_name_upper == 'ASN':
        if atom_name_upper in ['HD21', 'HD22', '1HD2', '2HD2']: # Common PDB names for ASN sidechain H
            return 'HD2'
    if res_name_upper == 'GLN':
        if atom_name_upper in ['HE21', 'HE22', '1HE2', '2HE2']: # Common PDB names for GLN sidechain H
            return 'HE2'

    # Methyl groups and other specific hydrogens from the map
    if res_name_upper in RELEVANT_HYDROGENS_MAP:
        # Check for specific PDB prefixes (e.g., HB, HG1, HD1)
        for pdb_prefix, casp_name in RELEVANT_HYDROGENS_MAP[res_name_upper].items():
            # Exact match (e.g. HB for ALA from some tools, or if OpenBabel names it simply)
            if atom_name_upper == pdb_prefix:
                return casp_name
            # Starts with prefix (e.g., HB1, HB2, HB3 for ALA)
            if atom_name_upper.startswith(pdb_prefix): # Catches HB1, HB2, HB3 etc.
                return casp_name
            # Check for cases like 1HB, 2HB (common from some tools)
            if len(atom_name_upper) > 1 and atom_name_upper[0].isdigit() and atom_name_upper[1:] == pdb_prefix:
                return casp_name
            # VAL/LEU/ILE methyls like HG11, HG12, HG13 -> HG1
            if res_name_upper in ['VAL', 'LEU', 'ILE']:
                 # HG11, HG12, HG13 -> HG1 for VAL
                 # HD11, HD12, HD13 -> HD1 for LEU
                 # Similar for HG2/HD2
                if atom_name_upper.startswith(pdb_prefix) and len(atom_name_upper) > len(pdb_prefix) and atom_name_upper[len(pdb_prefix)].isdigit():
                    return casp_name


    # Fallback for common methyl naming patterns if not caught above, ensuring it matches the map values
    # Example: ALA HB1, HB2, HB3 should all map to HB
    if res_name_upper == 'ALA' and atom_name_upper.startswith('HB'): # HB1, HB2, HB3
        return RELEVANT_HYDROGENS_MAP['ALA']['HB']
    # Example: VAL HG11, HG12, HG13 should map to HG1
    if res_name_upper == 'VAL':
        if atom_name_upper.startswith('HG1'): return RELEVANT_HYDROGENS_MAP['VAL']['HG1']
        if atom_name_upper.startswith('HG2'): return RELEVANT_HYDROGENS_MAP['VAL']['HG2']
    # Example: LEU HD11, HD12, HD13 should map to HD1
    if res_name_upper == 'LEU':
        if atom_name_upper.startswith('HD1'): return RELEVANT_HYDROGENS_MAP['LEU']['HD1']
        if atom_name_upper.startswith('HD2'): return RELEVANT_HYDROGENS_MAP['LEU']['HD2']
    # Example: ILE HD11, HD12, HD13 should map to HD1
    if res_name_upper == 'ILE':
        if atom_name_upper.startswith('HD1'): return RELEVANT_HYDROGENS_MAP['ILE']['HD1'] # Delta-methyl
        if atom_name_upper.startswith('HG2'): return RELEVANT_HYDROGENS_MAP['ILE']['HG2'] # Other methyl

    logger.debug(f"Atom {atom_name} on residue {residue_name} not mapped to CASP13 name.")
    return None

from .protein_residues import normal as RESIDUES
from Bio.SeqUtils import seq1, seq3

def PROCESS_RESIDUES(d):
    d['HIS'] = d['HIP']
    d = {key: val for key, val in d.items() if seq1(key) != 'X'}
    for key in d:
        atoms = d[key]['atoms']
        d[key] = {'CA': 'C'} | {key: val['symbol'] for key, val in atoms.items() if val['symbol'] != 'H' and key != 'CA'}
    return d
    
RESIDUES = PROCESS_RESIDUES(RESIDUES)

my_dir = f"/tmp/{os.getpid()}"
if not os.path.isdir(my_dir):
    os.mkdir(my_dir)

def pdb_to_npy(pdb_path, model_num=0, chain_id=None, seqres=None):
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        try:
            model = parser.get_structure('', pdb_path)[model_num]
            if chain_id is not None:
                chain = model.child_dict[chain_id]
            else:
                chain = model.child_list[0]
        except: logger.warning(f'Error opening PDB file {pdb_path}'); return

    coords = []
    seq = ''
    try:
        for resi in list(chain):
            if (resi.id[0] != ' ') or ('CA' not in resi.child_dict) or (resi.resname not in RESIDUES): continue
            co = np.zeros((14, 3)) * np.nan
            atoms = RESIDUES[resi.resname]
            seq += resi.resname
            for i, at in enumerate(atoms):
                try: co[i] = resi.child_dict[at].coord
                except: pass
            coords.append(co)
        coords = np.stack(coords)
        seq = seq1(seq)
    except: logger.warning(f'Error parsing chain {pdb_path}'); return
    if not seqres: return coords, seq

    coords_= np.zeros((len(seqres), 14, 3)) * np.nan
    alignment = pairwise2.align.globalxx(seq, seqres)[0]
    
    if '-' in alignment.seqB: 
        logger.warning(f'Alignment gaps {pdb_path}'); return
    mask = np.array([a == b for a, b in zip(alignment.seqA, alignment.seqB)])
    coords_[mask] = coords
    return coords_, mask

def tmscore(X_path, Y_path, molseq=None, lddt=True, lddt_start=1):
    if type(X_path) is not str:
        PDBFile(molseq).add(X_path).write(os.path.join(my_dir, 'X.pdb'))
        X_path = os.path.join(my_dir, 'X.pdb')
    if type(Y_path) is not str:
        PDBFile(molseq).add(Y_path).write(os.path.join(my_dir, 'Y.pdb'))
        Y_path = os.path.join(my_dir, 'Y.pdb')
    
    if not os.path.isabs(X_path): X_path = os.path.join(os.getcwd(), X_path)
    if not os.path.isabs(Y_path): Y_path = os.path.join(os.getcwd(), Y_path)
    
    out = subprocess.check_output(['TMscore', '-seq', Y_path, X_path], 
                    stderr=open('/dev/null', 'w'), cwd=my_dir)
    start = out.find(b'RMSD')
    end = out.find(b'rotation')
    out = out[start:end]
    
    rmsd, _, tm, _, gdt_ts, gdt_ha, _, _ = out.split(b'\n')
    
    rmsd = float(rmsd.split(b'=')[-1])
    tm = float(tm.split(b'=')[1].split()[0])
    gdt_ts = float(gdt_ts.split(b'=')[1].split()[0])
    gdt_ha = float(gdt_ha.split(b'=')[1].split()[0])
    
    if lddt:
        X_renum = os.path.join(my_dir, 'X_renum.pdb')
        renumber_pdb(molseq, X_path, X_renum, start=lddt_start)
        out = subprocess.check_output(['lddt', '-c', Y_path, X_renum],  # reference comes last
            stderr=open('/dev/null', 'w'), cwd=my_dir)
        for line in out.split(b'\n'):
            if b'Global LDDT score' in line:
                lddt = float(line.split(b':')[-1].strip())
        return {'rmsd': rmsd, 'tm': tm, 'gdt_ts': gdt_ts, 'gdt_ha': gdt_ha, 'lddt': lddt}
    else:
        return {'rmsd': rmsd, 'tm': tm, 'gdt_ts': gdt_ts, 'gdt_ha': gdt_ha}

    
class PDBFile:
    def __init__(self, molseq):
        self.molseq = molseq
        self.blocks = []
        self.chain = chain = Chain.Chain('A')
        j = 1
        for i, aa in enumerate(molseq):
            aa = Residue.Residue(
                id=(' ', i+1, ' '), 
                resname=seq3(aa).upper(), 
                segid='    '
            )
            for atom in RESIDUES[aa.resname.upper()]:
                at = Atom.Atom(
                    name=atom, coord=None, bfactor=0, occupancy=1, altloc=' ',
                    fullname=f' {atom} ', serial_number=j, element=RESIDUES[aa.resname][atom]
                )
                j += 1
                aa.add(at)
            chain.add(aa)
       
          
    def add(self, coords):
        if type(coords) is np.ndarray:
            coords = coords.astype(np.float64)
        elif type(coords) is torch.Tensor:
            coords = coords.cpu().double().numpy()
        
        for i, resi in enumerate(self.chain):
            atoms = RESIDUES[resi.resname]
            resi['CA'].coord = coords[i]

        k = len(self.chain)
        for i, resi in enumerate(self.chain):
            atoms = RESIDUES[resi.resname]
            for j, at in enumerate(atoms):
                if at != 'CA': 
                    try: resi[at].coord = coords[k] + resi['CA'].coord
                    except: resi[at].coord = (np.nan, np.nan, np.nan)
                    k+=1

        pdbio = PDBIO()
        pdbio.set_structure(self.chain)

        stringio = io.StringIO()
        stringio.close, close = lambda: None, stringio.close
        pdbio.save(stringio)
        block = stringio.getvalue().split('\n')[:-2]
        stringio.close = close; stringio.close()

        self.blocks.append(block)
        return self
        
    def clear(self):
        self.blocks = []
        return self
    
    def write(self, path=None, idx=None, reverse=False):
        is_first = True
        str_ = ''
        blocks = self.blocks[::-1] if reverse else self.blocks
        for block in ([blocks[idx]] if idx is not None else blocks):
            block = [line for line in block if 'nan' not in line]
            if not is_first:
                block = [line for line in block if 'CONECT' not in line]
            else:
                is_first = False
            str_ += 'MODEL\n'
            str_ += '\n'.join(block)
            str_ += '\nENDMDL\n'
        if not path:
            return str_
        with open(path, 'w') as f:
            f.write(str_)
            
def renumber_pdb(molseq, X_path, X_renum, start=1):
    assert type(molseq) == str
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        model = parser.get_structure('', X_path)[0]
    chain = model.child_list[0]
    polypeptide = Polypeptide.Polypeptide(chain.get_residues())
    seq = polypeptide.get_sequence()
    
    alignment = pairwise2.align.globalxx(seq, molseq)[0]
    assert '-' not in alignment.seqB
    numbering = [i+start for i, c in enumerate(alignment.seqA) if c != '-']
    
    for n, resi in enumerate(chain):
        resi.id = (' ', 10000+n, ' ')
    
    for n, resi in zip(numbering, chain):
        resi.id = (' ', n, ' ')
        
    pdbio = PDBIO()
    pdbio.set_structure(chain)
    pdbio.save(X_renum)

# The function format_noesy_outputs is no longer needed as its functionality
# has been integrated into generate_noesy_data.
# Consider removing it if it's not used elsewhere.
# def format_noesy_outputs(ground_truth_noes):
#     """
#     Formats the list of NOESY contact dictionaries into the specified string format.
#     Assigns a unique peak_id to each contact for now.
#     Noise and ambiguity will be handled separately.
#     """
#     formatted_noesy_strings = []
#     peak_id_counter = 1
#     for contact in ground_truth_noes:
#         res_num1 = contact['res_num1']
#         res_num2 = contact['res_num2']
#         atom_name1 = contact['atom_name1']
#         atom_name2 = contact['atom_name2']
#         distance = contact['distance']
#
#         formatted_string = f"{res_num1} {res_num2} {peak_id_counter} {distance:.1f} {atom_name1} {atom_name2}"
#         formatted_noesy_strings.append(formatted_string)
#         peak_id_counter += 1
#
#     logger.info(f"Formatted {len(formatted_noesy_strings)} NOESY contacts with unique peak IDs.")
#     return formatted_noesy_strings

import random # For noise generation

def generate_noesy_data(pdb_path, distance_cutoff=5.0, noise_fraction=0.1, false_positive_max_dist=10.0):
    """
    Generates NOESY-like data from a PDB file, including true and false positives.

    Args:
        pdb_path (str): Path to the input PDB file.
        distance_cutoff (float): Maximum distance for a true NOESY contact.
        noise_fraction (float): Fraction of false positives to add relative to true positives.
        false_positive_max_dist (float): Maximum distance for generating false positives.

    Returns:
        list: A list of strings, where each string represents a NOESY peak in the
              format "residueFrom residueTo peakID distance atomFrom atomTo".
              Returns an empty list if processing fails at any critical step.
    """
    logger.info(f"Starting NOESY data generation for {pdb_path} with cutoff {distance_cutoff}A.")
    hydrogenated_pdb_path = os.path.join("/tmp", os.path.basename(pdb_path).replace(".pdb", "_h.pdb"))
    os.makedirs("/tmp", exist_ok=True) # Ensure /tmp directory exists

    # Step 1: Add hydrogens using OpenBabel
    # It's crucial that OpenBabel is in the system PATH or obabel_path is correctly set.
    cmd = ['obabel', pdb_path, '-O', hydrogenated_pdb_path, '-h']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.error(f"OpenBabel failed for {pdb_path}. Return code: {result.returncode}. Error: {result.stderr.strip()}")
            if os.path.exists(hydrogenated_pdb_path): os.remove(hydrogenated_pdb_path)
            return []
        logger.info(f"OpenBabel successfully added hydrogens to {pdb_path}, output at {hydrogenated_pdb_path}")
    except FileNotFoundError:
        logger.error(f"OpenBabel command not found. Ensure OpenBabel is installed and in PATH.")
        if os.path.exists(hydrogenated_pdb_path): os.remove(hydrogenated_pdb_path)
        return []
    except Exception as e:
        logger.error(f"OpenBabel execution encountered an unexpected error for {pdb_path}: {e}")
        if os.path.exists(hydrogenated_pdb_path): os.remove(hydrogenated_pdb_path)
        return []

    # Step 2: Parse hydrogenated PDB and extract relevant hydrogens
    relevant_hydrogens = []
    structure = None
    # Use the globally defined parser
    current_parser = PDBParser() # Instantiate parser
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PDBExceptions.PDBConstructionWarning)
            # Ensure parser is instantiated
            structure = current_parser.get_structure("hydrogenated_protein", hydrogenated_pdb_path)

        if not structure:
            logger.error(f"BioPython PDBParser failed to parse {hydrogenated_pdb_path}. The file might be empty or corrupted.")
            if os.path.exists(hydrogenated_pdb_path): os.remove(hydrogenated_pdb_path)
            return []

        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != ' ' or Polypeptide.is_aa(residue.resname, standard=True) is False: # Skip HETATM, non-standard residues
                        continue
                    res_name = residue.resname.strip().upper()
                    res_num = residue.id[1]
                    for atom in residue:
                        atom_name = atom.name.strip() # Original atom name from PDB
                        # We rely on OpenBabel to correctly name hydrogens.
                        # We filter by element 'H' and then use our CASP13 mapping.
                        if atom.element == 'H': # Ensure it's a hydrogen atom
                            casp13_name = get_casp13_atom_name(res_name, atom_name)
                            if casp13_name:
                                relevant_hydrogens.append({
                                    'residue_number': res_num,
                                    'residue_name': res_name, # Keep for potential use in advanced noise gen
                                    'original_atom_name': atom_name, # Keep for debugging/logging
                                    'casp13_name': casp13_name,
                                    'coords': atom.coord
                                })

        if not relevant_hydrogens:
            logger.warning(f"No CASP13-relevant hydrogen atoms found in {hydrogenated_pdb_path} for {pdb_path}. Check PDB content and naming logic.")
            if os.path.exists(hydrogenated_pdb_path): os.remove(hydrogenated_pdb_path)
            return []
        logger.info(f"Extracted {len(relevant_hydrogens)} CASP13-relevant hydrogen atoms from {hydrogenated_pdb_path}.")

        # Step 3: Identify true NOESY contacts
        true_contacts = []
        for i in range(len(relevant_hydrogens)):
            for j in range(i + 1, len(relevant_hydrogens)): # Avoid self-pairs and duplicate pairs
                h1 = relevant_hydrogens[i]
                h2 = relevant_hydrogens[j]

                # Skip intra-residue contacts
                if h1['residue_number'] == h2['residue_number']:
                    continue

                distance = np.linalg.norm(h1['coords'] - h2['coords'])
                if distance <= distance_cutoff:
                    true_contacts.append({
                        'res_num1': h1['residue_number'],
                        'atom_name1': h1['casp13_name'],
                        'res_num2': h2['residue_number'],
                        'atom_name2': h2['casp13_name'],
                        'distance': float(distance),
                        'is_true': True
                    })
        logger.info(f"Identified {len(true_contacts)} true NOESY contacts within {distance_cutoff}A for {pdb_path}.")

        # Step 4: Generate false positives
        false_positives = []
        num_false_positives_to_generate = int(len(true_contacts) * noise_fraction)

        if num_false_positives_to_generate > 0 and len(relevant_hydrogens) > 1:
            attempts = 0
            max_attempts = num_false_positives_to_generate * 10 # Limit attempts to avoid infinite loops
            while len(false_positives) < num_false_positives_to_generate and attempts < max_attempts:
                attempts += 1
                idx1, idx2 = random.sample(range(len(relevant_hydrogens)), 2)
                h1 = relevant_hydrogens[idx1]
                h2 = relevant_hydrogens[idx2]

                if h1['residue_number'] == h2['residue_number']: # Skip intra-residue
                    continue

                # Check if this pair is already a true contact (based on atom names and residue numbers)
                is_already_true = False
                for tc in true_contacts:
                    if (tc['res_num1'] == h1['residue_number'] and tc['atom_name1'] == h1['casp13_name'] and \
                        tc['res_num2'] == h2['residue_number'] and tc['atom_name2'] == h2['casp13_name']) or \
                       (tc['res_num1'] == h2['residue_number'] and tc['atom_name1'] == h2['casp13_name'] and \
                        tc['res_num2'] == h1['residue_number'] and tc['atom_name2'] == h1['casp13_name']):
                        is_already_true = True
                        break
                if is_already_true:
                    continue

                # Generate a plausible but likely false distance
                # Could be random, or perturbed from actual distance if desired to be more deceptive
                actual_distance = np.linalg.norm(h1['coords'] - h2['coords'])
                if actual_distance <= distance_cutoff: # If it's a true contact missed by earlier logic or too close
                    false_distance = actual_distance + random.uniform(1.0, distance_cutoff) # Make it larger
                else:
                    # Generate a random distance that could be plausible, or slightly above cutoff
                    false_distance = random.uniform(distance_cutoff / 2, false_positive_max_dist)

                false_positives.append({
                    'res_num1': h1['residue_number'],
                    'atom_name1': h1['casp13_name'],
                    'res_num2': h2['residue_number'],
                    'atom_name2': h2['casp13_name'],
                    'distance': float(false_distance),
                    'is_true': False
                })
            logger.info(f"Generated {len(false_positives)} false positive NOESY contacts for {pdb_path}.")
        else:
            logger.info(f"Skipping false positive generation (num_false_positives_to_generate={num_false_positives_to_generate}, num_relevant_hydrogens={len(relevant_hydrogens)}).")

        # Step 5: Combine true and false contacts and assign unique peak IDs
        all_peaks = true_contacts + false_positives
        random.shuffle(all_peaks) # Shuffle to mix true and false positives

        formatted_noesy_strings = []
        for peak_id, peak_data in enumerate(all_peaks, 1): # Start peak IDs from 1
            # Output format: residueFrom residueTo peakID distance atomFrom atomTo
            # Ensure res_num1 < res_num2 or some consistent ordering if needed, but not specified.
            # For now, using the order as found/generated.
            formatted_string = (
                f"{peak_data['res_num1']} {peak_data['res_num2']} {peak_id} "
                f"{peak_data['distance']:.2f} {peak_data['atom_name1']} {peak_data['atom_name2']}"
            )
            formatted_noesy_strings.append(formatted_string)

        if not formatted_noesy_strings:
            logger.info(f"No NOESY data (true or false) generated for {pdb_path}.")
        else:
            logger.info(f"Successfully generated {len(formatted_noesy_strings)} total NOESY peaks for {pdb_path} ({len(true_contacts)} true, {len(false_positives)} false).")

        return formatted_noesy_strings

    except FileNotFoundError:
        logger.error(f"Hydrogenated PDB file not found for parsing: {hydrogenated_pdb_path}. This typically means OpenBabel failed to create it.")
        return []
    except PDBExceptions.PDBIOException as e:
        logger.error(f"BioPython PDBIOException while processing {hydrogenated_pdb_path} for {pdb_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing PDB {hydrogenated_pdb_path} for {pdb_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []
    finally:
        if os.path.exists(hydrogenated_pdb_path):
            try:
                os.remove(hydrogenated_pdb_path)
                logger.info(f"Removed temporary file {hydrogenated_pdb_path}")
            except OSError as e:
                logger.error(f"Error removing temporary file {hydrogenated_pdb_path}: {e}")