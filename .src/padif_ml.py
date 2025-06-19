import os
import gc
import ast
import glob
import time
import random
import shutil
import warnings
import tempfile
import deepchem
import argparse
import subprocess
import numpy as np
import pandas as pd
import datamol as dm
from sys import argv
from joblib import Parallel, delayed
from docking_tools import chembl_mols, prep_ligand_from_smiles, gold_config, dock_mol, gen_scaffold, protein_prepare
from fp_extract import padif_dataframe
from split_datasets import to_split
from ml_training import padif_train
from metrics import predictions_models, figures_from_metrics, test_predictions
from residue_atom_info import atom_interaction_info, atoms_by_activity, residue_interactions, write_chimera_commands
"""
Starting calculations
"""
### Initial parameters
njobs = int(20)
# njobs = os.cpu_count()
warnings.filterwarnings("ignore")

### time
start_cpu = time.process_time()
start_clock = time.time()

def chembl_download(chembl_code):

    ### Download the molecules and create datframes from ChEMBL
    dfActivities, targetName = chembl_mols(chembl_code)

    ### Delete bad strings in target name
    unaceptable_strings = ["/", "(", ")", ",", ";", ".", " "]
    for string in unaceptable_strings:
        targetName = targetName.replace(string, "_")

    return dfActivities, targetName

def smiles_standardization(smiles):
    try:
        mol = dm.to_mol(smiles, ordered=True)
        mol = dm.fix_mol(mol, largest_only=True)
        mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
        mol = dm.standardize_mol(
            mol, disconnect_metals=True, normalize=True, reionize=False, uncharge=False, stereo=False
        )
        smiles = dm.to_smiles(mol, isomeric=True, canonical=False)
    except:
        smiles = 'Error in smiles'
    
    return smiles

def protein_process(protein, ligand_id, path_target, target_name):    

    ### Open and prepare protein 
    prot1 = protein

    ### preprare protein and ligand
    protein_prepare(
        protein=prot1,
        path=path_target,
        target_name=target_name,
        save_ligand=True,
        ligand_id=ligand_id,
        multiple_chain=False,
        chain_to_work='A',
        remove_metals=False
    )

def active_docking(ligands, path_target, target_name, temp, type_binding_site='ligand', coordinates=None, residues=None, size=8):
    
    ### Make a temp folder and prepare ligands
    act_dir = tempfile.mkdtemp(suffix=None,prefix='act_',dir=temp)
    ligands['act_dir'] = [act_dir]*len(ligands)
    tasks = ligands.apply(lambda row: (row['smiles'], row['id'], row['act_dir']), axis=1)
    Parallel(n_jobs=-1)(delayed(prep_ligand_from_smiles)(*task) for task in tasks)

    ### create the gold config for docking
    if type_binding_site == 'ligand':
        gold_config(protein=f"{path_target}/{target_name}_prep.mol2", 
                    ref_ligand=f"{path_target}/Ligand_{target_name}.mol2",
                    gold_name=f'{target_name}_gold',
                    size=size,
                    path=path_target,
                    ccdc_older_2020 = True,
                    ccdc_path='/mnt/ccdc')
    elif type_binding_site == 'coordinates':
        if coordinates == None:
            print('please add coordinates')
        else:
            gold_config(protein=f"{path_target}/{target_name}_prep.mol2", 
                        type_of_binding_site= "coordinates",
                        size=size,
                        coordinates=coordinates,
                        gold_name=f'{target_name}_gold',
                        path=path_target,
                        ccdc_older_2020 = True,
                        ccdc_path='/mnt/ccdc')
    elif type_binding_site == 'residues':
        if residues == None:
            print('please add residues list')
        else:
            gold_config(protein=f"{path_target}/{target_name}_prep.mol2", 
                        type_of_binding_site= "residues",
                        size=size,
                        residues=residues,
                        gold_name=f'{target_name}_gold',
                        path=path_target,
                        ccdc_older_2020 = True,
                        ccdc_path='/mnt/ccdc')

    ### Dock the Active compounds
    to_dock = ligands[['id', 'act_dir']]
    to_dock['gold_file'], to_dock['num_sln'] = [f'{path_target}/{target_name}_gold.conf']*len(to_dock), [10]*len(to_dock)
    tasks_dock = to_dock.apply(lambda row: (row['gold_file'], row['id'], row['act_dir'], row['num_sln']), axis=1)
    Parallel(n_jobs=-1)(delayed(dock_mol)(*task) for task in tasks_dock)

    ### Create active and inactive directories and save docking files
    act_f = os.path.join(path_target, "Actives")
    os.makedirs(act_f, exist_ok=True)

    for filename in glob.glob(act_dir + f"/*_sln.sdf"):
        shutil.copy(filename, act_f)

    ### Copy plp_protein to principal folder
    shutil.copy(f"{act_dir}/plp_protein.mol2", path_target)  

    return act_f  

def decoy_docking(ligands, temp, path_target, target_name, type_binding_site='ligand', coordinates=None, residues=None, size=8):

    ### Calculate scaffolds for active compounds
    scaffolds = Parallel(n_jobs=-1)(delayed(gen_scaffold)(smi) for smi in ligands.smiles.values)
    ligands['scf'] = scaffolds
    ligands = ligands[(ligands['scf'] != 'Error-Smiles') & (ligands['scf'] != 'Something else')]    

    ### Select different compounds as decoys using scaffolds
    decoys_df = pd.read_csv('/code/src/DCM_prepared.csv', sep=',')  
    bad_id_dec = pd.merge(decoys_df, ligands, on="scf")["id_x"].unique().tolist()
    decoys_df_f = decoys_df[~decoys_df.id.isin(bad_id_dec)]
    decoys = decoys_df.iloc[random.sample(range(len(decoys_df_f)), len(ligands)*9)] 

    ### make a temp folder and prepare decoys
    dec_dir = tempfile.mkdtemp(suffix=None,prefix='ina_',dir=temp)
    decoys['ina_dir'] = [dec_dir]*len(decoys)
    tasks_decoys = decoys.apply(lambda row: (row['smiles'], row['id'], row['ina_dir']), axis=1)
    Parallel(n_jobs=-1)(delayed(prep_ligand_from_smiles)(*task) for task in tasks_decoys) 

    ### create the gold config for docking
    if type_binding_site == 'ligand':
        gold_config(protein=f"{path_target}/{target_name}_prep.mol2", 
                    ref_ligand=f"{path_target}/Ligand_{target_name}.mol2",
                    gold_name=f'{target_name}_gold_decoys',
                    size=size,
                    path=path_target,
                    div_solutions= True,
                    ccdc_older_2020 = True,
                    ccdc_path='/mnt/ccdc')
    elif type_binding_site == 'coordinates':
        if coordinates == None:
            print('please add coordinates')
        else:
            gold_config(protein=f"{path_target}/{target_name}_prep.mol2", 
                        type_of_binding_site= "coordinates",
                        size=size,
                        coordinates=coordinates,
                        gold_name=f'{target_name}_gold_decoys',
                        path=path_target,
                        div_solutions= True,
                        ccdc_older_2020 = True,
                        ccdc_path='/mnt/ccdc')
    elif type_binding_site == 'residues':
        if residues == None:
            print('please add residues list')
        else:
            gold_config(protein=f"{path_target}/{target_name}_prep.mol2", 
                        type_of_binding_site= "residues",
                        size=size,
                        residues=residues,
                        gold_name=f'{target_name}_gold_decoys',
                        path=path_target,
                        div_solutions= True,
                        ccdc_older_2020 = True,
                        ccdc_path='/mnt/ccdc')

    ### Dock the Active compounds
    to_dock_decoys = decoys[['id', 'ina_dir']]
    to_dock_decoys['gold_file'], to_dock_decoys['num_sln'] = [f'{path_target}/{target_name}_gold_decoys.conf']*len(to_dock_decoys), [10]*len(to_dock_decoys)
    tasks_dock_decoys = to_dock_decoys.apply(lambda row: (row['gold_file'], row['id'], row['ina_dir'], row['num_sln']), axis=1)
    Parallel(n_jobs=-1)(delayed(dock_mol)(*task) for task in tasks_dock_decoys)   

    ### Create decoys directory, save docking files and copy protein
    dec_f = os.path.join(path_target, "Decoys")
    os.makedirs(dec_f, exist_ok=True)   

    for filename in glob.glob(dec_dir + f"/*_sln.sdf"):
        shutil.copy(filename, dec_f)    

    shutil.copy(f"{dec_dir}/plp_protein.mol2", path_target)

    return decoys, dec_f

def test_docking(test_df, ligands, temp, path_target, target_name):

    ### Select different compounds as decoys using scaffolds
    bad_id_test = pd.merge(test_df, ligands, on="smiles")["id_x"].unique().tolist()
    test = test_df[~test_df.id.isin(bad_id_test)]

    ### make a temp folder and prepare test
    test_dir = tempfile.mkdtemp(suffix=None,prefix='test_',dir=temp)
    test['test_dir'] = [test_dir]*len(test)
    tasks_test = test.apply(lambda row: (row['smiles'], row['id'], row['test_dir']), axis=1)
    results = Parallel(n_jobs=-1)(delayed(prep_ligand_from_smiles)(*task) for task in tasks_test) 

    ### Dock the Active compounds
    to_dock_test = test[['id', 'test_dir']]
    to_dock_test['gold_file'], to_dock_test['num_sln'] = [f'{path_target}/{target_name}_gold.conf']*len(to_dock_test), [10]*len(to_dock_test)
    tasks_dock_test = to_dock_test.apply(lambda row: (row['gold_file'], row['id'], row['test_dir'], row['num_sln']), axis=1)
    results_dock_test = Parallel(n_jobs=-1)(delayed(dock_mol)(*task) for task in tasks_dock_test)   

    ### Create test directory, save docking files and copy protein
    test_f = os.path.join(path_target, "test_mols")
    os.makedirs(test_f, exist_ok=True)   

    for filename in glob.glob(test_dir + f"/*_sln.sdf"):
        shutil.copy(filename, test_f)    

    shutil.copy(f"{test_dir}/plp_protein.mol2", path_target)

    return test_f, bad_id_test

def split_datasets(splitters, path_target, target_name):

    data_to_model = os.path.join(path_target, 'data_to_model')
    os.makedirs(data_to_model, exist_ok=True)
    ### Split with all types of splitters
    for splitter in splitters:
        spl_folder = os.path.join(data_to_model, splitter)
        os.makedirs(spl_folder, exist_ok=True)
        path = f'{data_to_model}/{splitter}'
        to_split(target=target_name, padif_folder=path_target, path_to_work=path, method=splitter)
   

def main(args):

    ### Parser
    parser = argparse.ArgumentParser(description='Information to run PADIF-app')

    parser.add_argument('chembl_code', help='ChEMBL ID associated with the target to work')
    parser.add_argument('protein_file', help='Protein file in PDB format to dock all of compounds')
    parser.add_argument('ligand_id', help='ID of ligand used to define grid for docking')
    parser.add_argument('test_molecules', help='CSV file with ID and SMILES for molecules to test', default=None)

    parsed_args = parser.parse_args(args)

    ### Download the ligands from chembl and change columns names
    ligands, target_name = chembl_download(parsed_args.chembl_code)
    
    ### Folder to work
    path_target = os.path.join(f"/code/work/", target_name)
    os.makedirs(path_target, exist_ok=True)

    ligands = ligands[['molecule_chembl_id', 'canonical_smiles']].rename(columns={'molecule_chembl_id':'id', 'canonical_smiles':'smiles'})
    
    print(
        f'''    
        ********************************************************************************
        
        Starting to work with {target_name} docking process
        
        ********************************************************************************
        '''
    )

    ### SMILES standardization
    ligands['smiles'] = Parallel(n_jobs=-1)(delayed(smiles_standardization)(smi) for smi in ligands.smiles)

    print(f"""
        Number of smiles that cannot be standardized in chembl ligands: {len(ligands[ligands['smiles'] == 'Error in smiles'])}
    """
    )

    ligands = ligands[ligands['smiles'] != 'Error in smiles']

    ligands.to_csv(f'{path_target}/{target_name}_chembl_mols.csv', index=False)
    protein_process(f'/code/work/{parsed_args.protein_file}', parsed_args.ligand_id, path_target, target_name)

    ### Create temporary folder to work
    sysTemp = tempfile.gettempdir()
    temp = os.path.join(sysTemp,'temp')
    #You must make sure temp exists
    if not os.path.exists(temp):
        os.makedirs(temp)

    coordinates = (-28.577,4.072,-8.874)        
    active_folder = active_docking(ligands, path_target, target_name, type_binding_site='coordinates',coordinates=coordinates, size=8, temp=temp)
    active_padif = padif_dataframe(path_solutions=active_folder, path=path_target)

    decoys, decoys_folder = decoy_docking(ligands, temp, path_target, target_name, type_binding_site='coordinates',coordinates=coordinates, size=8)
    decoys.to_csv(f'{path_target}/decoys_used.csv', index=False)
    decoys_padif = padif_dataframe(path_solutions=decoys_folder, padif_type='decoys', padif_ref=active_padif, path=path_target)

    ### create a single padif and concatenate
    ligand_info = ligands[['id', 'smiles']]
    active_padif[['id', 'pose']] = active_padif['id'].str.split('_', expand=True)
    active_padif = active_padif.merge(ligand_info, on='id')
    active_padif['id'] = active_padif['id'].str.cat(active_padif.pose, sep='_')
    active_padif = active_padif.drop('pose', axis=1)
    decoys_info = decoys[['id', 'smiles']]
    decoys_padif[['id', 'pose']] = decoys_padif['id'].str.split('_', expand=True)
    decoys_padif = decoys_padif.merge(decoys_info, on='id')
    decoys_padif['id'] = decoys_padif['id'].str.cat(decoys_padif.pose, sep='_')
    decoys_padif = decoys_padif.drop('pose', axis=1)
    ### concat the list and save PADIF file
    final_padif = pd.concat([active_padif, decoys_padif], ignore_index=True)
    final_padif = final_padif.fillna(0.0)
    ### Sort columns and put in the end of dataframe for better visualization
    to_move = ['score', 'id', 'smiles','activity']
    order_1 =[col for col in final_padif.columns if col not in to_move] + to_move
    final_padif = final_padif[order_1]
    final_padif.to_csv(f'{path_target}/{target_name}_PADIF.csv', sep=',', index=False)

    gc.collect()

    splitters = ['scaffold', 'fingerprint', 'random']

    split_datasets(splitters, path_target, target_name)

    ### Train models
    padif_train(target_name, splitters, path_target)

    ### Save figures and statistics
    metrics_df, predictions_df = predictions_models(splitters, path_target)
    figures_from_metrics(predictions_df, metrics_df, path_target)

    metrics_df.to_csv(f'{path_target}/metrics_of_models.csv', sep=',', index=False)
    predictions_df.to_csv(f'{path_target}/predictions_of_models.csv', sep=',', index=False)

    ### Test compounds
    if not os.path.isfile(f'/code/work/{parsed_args.test_molecules}'):
        print(
            """
            Test file missing, padif app is end 
            """
        )
        ### Remove Bad files
        gc.collect()
        shutil.rmtree(temp)

        sys.exit(1)
    else:
        test_df = pd.read_csv(f'/code/work/{parsed_args.test_molecules}')
        test_df['smiles'] = Parallel(n_jobs=-1)(delayed(smiles_standardization)(smi) for smi in test_df.smiles)
        other = test_df[test_df['smiles'] == 'Error in smiles']
        test_df = test_df[test_df['smiles'] != 'Error in smiles']

        ### Filter by molecular weight
        def molwt(smiles):
            mol = dm.to_mol(smiles)
            return dm.descriptors.mw(mol)

        test_df['mol_weight'] = Parallel(n_jobs=-1)(delayed(molwt)(smi) for smi in test_df.smiles)

        test_df = test_df.loc[(test_df["mol_weight"] >= 180.0) &
                            (test_df["mol_weight"] <= 600.0)]

        print(f"""
            Number of smiles that cannot be standardized in tested compounds: {len(test_df[test_df['smiles'] == 'Error in smiles'])}
        """
        )
        test_df.to_csv(f'{path_target}/test_compounds.csv', index=False)

        print(f"""
            Number of compounds to dock: {len(test_df)}
        """
        )

        test_folder, bad_ids = test_docking(test_df, ligands, temp, path_target, target_name)
        
        ### report bad tested molecules
        bad_test = test_df[test_df.id.isin(bad_ids)]
        pd.concat([bad_test, other], axis=0).reset_index(drop=True)
        bad_test.to_csv(f'{path_target}/bad_tested_compounds.csv', index=False)

        ### Extract PADIF
        test_padif = padif_dataframe(path_solutions=decoys_folder, padif_type='test', padif_ref=final_padif, path=path_target)
        test_padif = test_padif[~test_padif.id.isin(bad_ids)]
        test_padif.to_csv(f'{path_target}/test_PADIF.csv', index=False)
        prediction = test_predictions(padif, test_padif, path_target, splitters)
        prediction.to_csv(f'{path_target}/test_predictions.csv', index=False)            


    ### Remove Bad files
    gc.collect()
    shutil.rmtree(temp)

if __name__ == '__main__':
    main(argv[1:])


