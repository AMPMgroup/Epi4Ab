import os
from tqdm import tqdm
from Bio.PDB.PDBIO import Select, PDBIO
from Bio.PDB.MMCIFParser import FastMMCIFParser
import json
from pathlib import Path

class RecSelect(Select):
    def __init__(self, cond_lst, rare_aa):
        self.cond = cond_lst
        self.rare_aa = rare_aa

    def accept_chain(self, chain):
        if chain.get_id() in self.cond:
            return True
        else:
            return False
            
    def accept_residue(self,residue):
        if (residue.get_id()[0] == ' ') & (residue.get_resname() != 'UNK'):
            return True
        elif (residue.get_id()[0][2:] in self.rare_aa):
            residue.resname = self.rare_aa[residue.resname]
            residue.id = tuple([' ',residue.id[1],residue.id[2]])
            return True
        else:
            return False

    def accept_atom(self, atom):
        if (atom.get_altloc() == ' '):
            return True
        elif (atom.get_altloc() == 'A'):
            atom.set_altloc(' ')
            return True
        else:
            return False

def filter_pdb(pdb_df, logging):
    parser = FastMMCIFParser(auth_residues = False, QUIET = True)
    io = PDBIO()
    with open(Path(__file__).parent / 'amino_acid_rare.json', 'r') as f:
        rare_aa = json.load(f)
    for pdbId in tqdm(pdb_df.pdbID, desc = 'Filter structure', unit='pdb'):
        # Filter PDB structure
        pdbSeries = pdb_df[pdb_df.pdbID == pdbId]
        pdb_id_path = os.path.join(logging.directory_data, pdbId)
        pdbId = pdbSeries.pdbID.item()
        pdb = pdbSeries.pdb.item()
        AgChain = pdbSeries.antigen.item()
        try:
            structure = parser.get_structure(pdbId, os.path.join(pdb_id_path, f'{pdb}.cif'))
                
            io.set_structure(structure)

            io.save(os.path.join(pdb_id_path, 'lig.pdb'), RecSelect(AgChain,rare_aa))
        except:
            print(pdbId)
            logging.error_extract_structure.append(pdbId)
        
    if not logging.error_extract_structure:
        logging.message += '''
All pdb structures have been extracted successfully.'''
    else:
        logging.message += f'''
Not extract structure pdb(s): {logging.error_extract_structure}'''