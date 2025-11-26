load /mnt/edward_data/ampm_project/vsc_module/Epi4Ab/output_preprocess/processed_data/1n8z_BAC/lig.pdb, mol 
sele near10A, resi 607 and chain C around 10 
iterate near10A and name CA, print("##",resn,resi,chain,cmd.distance("resi 607 and name CA and chain C","resi %s and name CA and chain %s" %(resi,chain))) 
