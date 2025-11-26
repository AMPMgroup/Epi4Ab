import os
from tqdm import tqdm
import pandas as pd
import MDAnalysis as mda

def backbone_angle_nan(series):
    psi = pd.isnull(series.psi)
    phi = pd.isnull(series.phi)
    omega = pd.isnull(series.omega)
    if psi & phi & omega:
        return 1
    else:
        return 0

def extract_angle(pdb_df, logging):
    for pdbId in tqdm(pdb_df.pdbID, desc = 'Extract angle', unit='pdb'): 
        pdb_id_path = os.path.join(logging.directory_data, pdbId)
        filePath = os.path.join(pdb_id_path,'lig.pdb')
        resList = mda.Universe(filePath).residues
        finalAngle = []
        
        for ind, res in enumerate(resList):
            try:
                if ind == len(resList) - 1:
                    psi = None
                    phi = None
                    omega = None
                elif res.segid != resList[ind+1].segid:
                    psi = None
                    phi = None
                    omega = None
                else:
                    phi = resList[ind+1].phi_selection()
                    if phi is not None:
                        phi = phi.dihedral.value()
                    psi = res.psi_selection()
                    if psi is not None:
                        psi = psi.dihedral.value()
                    omega = res.omega_selection()
                    if omega is not None:
                        omega = omega.dihedral.value()
                chi1 = res.chi1_selection()
                if chi1 is not None:
                    chi1 = chi1.dihedral.value()
            except:
                logging.error_angle.append(pdbId)
            finalAngle.append([pdbId, res.segid, res.resname, res.resid , psi, phi, omega, chi1])
        finalAngle = pd.DataFrame(finalAngle, columns = ['pdbId', 'chainId', 'resName', 'resId','psi', 'phi', 'omega', 'chi'])
        finalAngle['angleNan'] = finalAngle.apply(lambda x: backbone_angle_nan(x), axis = 1)
        finalAngle['chiNan'] = finalAngle.apply(lambda x: 1 if pd.isnull(x.chi) else 0, axis = 1)
        finalAngle['psi'] = finalAngle['psi'].fillna(0)
        finalAngle['phi'] = finalAngle['phi'].fillna(0)
        finalAngle['omega'] = finalAngle['omega'].fillna(0)
        finalAngle['chi'] = finalAngle['chi'].fillna(0)

        outDir = os.path.join(pdb_id_path, 'angle')
        if not os.path.exists(outDir):
            os.mkdir(outDir)
        finalAngle[['pdbId','chainId','resName']] = finalAngle[['pdbId','chainId','resName']].astype('string')
        finalAngle.to_parquet(os.path.join(outDir, 'angle_result.parquet'))
        # finalAngle.to_csv(os.path.join(outDir, 'angle_result.txt'), index=None)

    if not logging.error_angle:
        logging.message += '''
All pdb angle have been extracted successfully.'''
    else:
        logging.message += f'''
No angle pdb(s): {logging.error_angle}'''

        
