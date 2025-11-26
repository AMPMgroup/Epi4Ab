import os
from Bio.PDB import PDBParser
from tqdm import tqdm
import math

# def extract_rg(pdb_df, logging):
#     parser = PDBParser(QUIET=True)
#     for pdb_id in tqdm(pdb_df.pdbID, desc='Extract Radius of Gyration', unit='pdb'):
#         pdb_path = os.path.join(logging.directory_data, pdb_id)
#         if logging.process_relaxed:
#             lig_file = os.path.join(logging.directory_data, pdb_id[:-3], 'lig_re.pdb')
#         else:
#             lig_file = os.path.join(pdb_path, 'lig.pdb')

'''
https://github.com/sarisabban/Rg/blob/master/Rg.py
'''

def extract_rg(structure):
    coord = []
    mass = []
    for chain in structure[0]:
        for res in chain:
            for atom in res:
                if atom.get_name()[0] == 'C':
                    mass.append(12.0107)
                elif atom.get_name()[0] == 'O':
                    mass.append(15.9994)
                elif atom.get_name()[0] == 'N':
                    mass.append(14.0067)
                elif atom.get_name()[0] == 'S':
                    mass.append(32.065)
                coord.append(atom.get_coord())
    xm = [(m*i, m*j, m*k) for (i, j, k), m in zip(coord, mass)]
    tmass = sum(mass)
    rr = sum(mi*i + mj*j + mk*k for (i, j, k), (mi, mj, mk) in zip(coord, xm))
    mm = sum((sum(i) / tmass)**2 for i in zip(*xm))
    return math.sqrt(rr / tmass-mm)
    