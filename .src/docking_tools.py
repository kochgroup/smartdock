"""
With this function, download molecules from CHEMBL, preprare them,
write GOLD config file, made the molecular docking and extract generic scaffold
is possible. 
IMPORTANT: this only runs with CCDC enviroment, please install the before run
"""
import pandas as pd
import datamol as dm
from math import log
from joblib import Parallel, delayed
from ccdc.docking import Docker
from ccdc.io import MoleculeReader, MoleculeWriter
from ccdc.entry import Entry
from ccdc.molecule import Molecule
from ccdc.protein import Protein
from ccdc.conformer import ConformerGenerator
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric
from chembl_webresource_client.new_client import new_client 

def chembl_mols(chembl_id):
    """
    Download all molecules related with the specific target in chembl, selecting only the unique compunds with 
    activities reported in "IC50","Ki","EC50","Kd" and each molecular weight is between 180 and 600 daltons

    Parameters
    ----------
    chembl_id: str
        ChEMBL id code for a specific target
    
    Return
    ------
    dfActivities: Pandas dataframe
        A dataframe with molecules and other information related with this compunds
    targetName: str
        The common name reported in ChEMBL        
    """
    activity =  new_client.activity
    target = new_client.target
    listOfActivities = activity.filter(target_chembl_id=chembl_id).filter(standard_units="nM").only(
                "canonical_smiles", "molecule_chembl_id", "pchembl_value", 
                "standard_units", "standard_value", "standard_type"
            )
    if not listOfActivities:
        print("Error in ChEMBL ID")
    else:

        targetInfo = target.filter(target_chembl_id=chembl_id).only(
                    "pref_name", "target_type"
                    )
        targetData = pd.DataFrame(targetInfo)
        targetName = targetData["pref_name"][0] 

        ### Do a dataframe and calculate pIC50
        dfActivities = pd.DataFrame(listOfActivities)
        dfActivities = dfActivities[dfActivities['standard_type'].isin(["IC50","Ki","EC50","Kd"])]
        dfActivities = dfActivities.astype({"standard_value":float})
        dfActivities = dfActivities[dfActivities["standard_value"] > 0]
        dfActivities["pIC\u2085\u2080"] = [-log(i / 10**9, 10) for i in dfActivities["standard_value"]]

        ### Sort by pIC50 and delete the repeated molecules

        dfActivities = dfActivities.sort_values(by=["pIC\u2085\u2080"], ascending=False)
        dfActivities = dfActivities.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)
        dfActivities = dfActivities[dfActivities['pIC\u2085\u2080'] >= 5]
        
        ### Select molecules only for molecular weight required
        
        dfActivities['mol_weight'] = [
            dm.descriptors.mw(mol) if mol is not None else 0
            for mol in [dm.to_mol(smi) for smi in dfActivities.canonical_smiles]
        ]

        dfActivities = dfActivities.loc[(dfActivities["mol_weight"] >= 180.0) &
                                        (dfActivities["mol_weight"] <= 600.0), 
                                        ["canonical_smiles", "molecule_chembl_id", "pchembl_value", 
                                        "standard_units", "standard_value", "standard_type",
                                        "mol_weight","pIC\u2085\u2080"]]


        return dfActivities, targetName 

def prep_ligand_from_smiles(smiles, id, dir):
    
    """
    Prepare molecules for docking using GOLD ligandPreparation function from a smiles

    Parameters
    ----------
    smiles: str
        Smiles to prepare
    id: str
        Name or id for each molecule
    dir: str
        Name of directory to save the file
    
    Return
    ------
    prep_lig: mol2 file
        Molecular file for the prepared structure
    """
    try:
        ### Read the molecules with rdkit 
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        ### Calculate the Conformers and enegies for each one
        conformer = AllChem.EmbedMultipleConfs(
            mol, 
            numConfs=100, 
            pruneRmsThresh=1, 
            numThreads=-1, 
            useRandomCoords=True, 
            randomSeed=1590
        )
        energies = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=200)
        ### Sort the conformers and select the conformer with the best energy
        sorted_conformers = sorted(range(len(conformer)), key=lambda i: energies[i][1])
        ### Pass the conformer to SDF mol and read with CCDC
        conf_mol = Chem.MolToMolBlock(mol, confId=sorted_conformers[0])
        ccdc_mol = Molecule.from_string(conf_mol, format="sdf")
        ### Prepare the compound for docking 
        ligand_prep = Docker.LigandPreparation()
        ligand_prep.settings.protonate = True
        ligand_prep.settings.standardise_bond_types = True
        prep_lig = ligand_prep.prepare(Entry.from_molecule(ccdc_mol))
        ### Write molecule 
        with MoleculeWriter(f"{dir}/{id}.mol2") as mol_writer:
            mol_writer.write(prep_lig.molecule)
    except:
        print(f'''
            {id} cannot be processed {smiles}
        ''')
    

def gold_config(protein, path, type_of_binding_site= "ligand", ref_ligand=None,  residues_numbers=None,
                coordinates = None, gold_name = "gold", size = 8, div_solutions = False, ccdc_older_2020 = False,
                ccdc_path = None):
    """
    GOLD configuration file for molecular docking

    Parameters
    ----------
    protein: pdb or mol2 file
        Protein file for make the docking
    path: str
        Directory or path to save the files
    type_of_binding_site: ligand, coordinates, residues
        Center of binding site for the molecular docking
    ref_ligand: pdb or mol2 file
        Reference ligand for molecular docking
    residue: list
        List of residue numbers for center the molecular docking
    coordinates: set
        Set of coordinates (X,Y,Z) for center the molecular docking
    gold_name: str
        Name for GOLD config file
    size: float
        Size of the grid for docking
    div_solutions: Boolean
        Boolean operator for use diverse solutions option from GOLD docking, (clusters = 1, RMSD= 1.5)
    ccdc_older_2020: Boolean
        Boolean operator for define a older version of CCDC software, less than 2020 version
    ccdc_path: str
        Path for CCDC software

    Return
    ------
    Gold_config: config file
        GOLD docking config file for make docking
    """        
    ### call the functions to dock
    docker = Docker()
    settings = docker.settings
    
    ### call protein and native ligand for select the binding site
    prot_1 = protein
    settings.add_protein_file(prot_1)
    if type_of_binding_site == 'ligand':
        native_ligand = ref_ligand
        native_ligand_mol = MoleculeReader(native_ligand)[0]
    prot_dock = settings.proteins[0]

    ### Select parameters to dock: binding site, fitness_function, autoscale, and others
    if type_of_binding_site == 'ligand':
        settings.binding_site = settings.BindingSiteFromLigand(prot_dock, native_ligand_mol, size)
    if type_of_binding_site == 'residues':
        index = [i for i, res in enumerate(prot_dock.residues) for res_number in residues_numbers if str(res).endswith(f'{res_number})')]
        res = [prot_dock.residues[val] for val in index]
        settings.binding_site = settings.BindingSiteFromListOfResidues(prot_dock, res)
    elif type_of_binding_site == 'coordinates':                                                                                                             
        settings.binding_site = settings.BindingSiteFromPoint(prot_dock, origin=coordinates, distance=size)
    ### Select other options for molecular docking
    settings.fitness_function = "PLP"
    settings.autoscale = 200
    settings.early_termination = False
    settings.write_options = "NO_GOLD_LOG_FILE NO_LOG_FILES NO_LINK_FILES NO_RNK_FILES NO_BESTRANKING_LST_FILE NO_GOLD_PROTEIN_MOL2_FILE NO_LGFNAME_FILE NO_PID_FILE NO_SEED_LOG_FILE NO_GOLD_ERR_FILE NO_FIT_PTS_FILES NO_GOLD_LIGAND_MOL2_FILE"
    settings.flip_amide_bonds = True
    settings.flip_pyramidal_nitrogen = True
    settings.flip_free_corners = True
    settings.save_binding_site_atoms = True 
    ### Use older version of CCDC software 
    if ccdc_older_2020 == True:
        settings.torsion_distribution_file = f"{ccdc_path}/CSDS2020/Discovery_2020/GOLD/gold/gold.tordist"
    if div_solutions == True:
        settings.diverse_solutions = (True, 1, 1.5)

    ### save the configuration file to modify
    Docker.Settings.write(settings,f"{path}/{gold_name}.conf")

    ### Add to config file "per_atom_scores"
    with open(f"{path}/{gold_name}.conf", "r") as inFile:
        lines = inFile.readlines()

    new_lines = []
    for line in lines:
        new_lines.append(line)

        if line.strip() == 'save_protein_torsions = 1':
            new_lines.extend([
                'concatenated_output = ligand.sdf\n',
                "output_file_format = MACCS\n",
                "per_atom_scores = 1\n"
            ])
    
    # Write the updated content back to the file
    with open(f"{path}/{gold_name}.conf", "w") as file:
        file.writelines(new_lines)

### Docking function

def dock_mol(confFile, id, dir, num_sln = 10):
    """
    Docking function 

    Parameters
    ----------
    confFile: str
        Name of the GOLD config file
    id: str
        Name or id for each molecule
    dir: str
        Name of the folder for save dockings
    num_sln: int
        Number of docking solutions
    
    Return
    ------
    docking_sln_file: sdf file 
        Docking solution file in sdf
    """
    conf = confFile
    settings = Docker.Settings.from_file(conf)
    ligand = f"{id}.mol2"
    settings.add_ligand_file(ligand, num_sln)
    settings.output_directory = dir
    settings.output_file = f"{id}_sln.sdf"
    docker = Docker(settings = settings).dock(f"{dir}/{id}.conf")

def gen_scaffold(smiles):
    """
    Create a Generic scaffold from smiles

    Parameters
    ----------
    smiles: str
        Smiles to extract generic scaffold
    
    Return
    ------
    scf: str
        Smiles for each scaffold, or errors in this calculation
    """
    try:
        ### Try to open and convert smiles to rdkit molecules
        mol = Chem.MolFromSmiles(smiles)
        if mol == None:
            return "Error-Smiles"
        else:
        ### Extract generic scaffold from molecules
            scf = MakeScaffoldGeneric(mol)
            return Chem.MolToSmiles(scf)
    except:
        return "Something else"

def protein_prepare(protein, path, target_name, save_ligand=True, ligand_id=None ,multiple_chain=False, chain_to_work='A', remove_metals=True):
    """
    Prepare protein for GOLD docking

    Parameters
    ----------
    protein: str
        File with protein in PDB format
    path: str
        Folder to save protein and ligand files
    target_name: str
        Name to follow the protein
    save_ligand: bool
        If is true, the ligand asociated to PDB file is save it
    ligand_id: str
        Id for ligand in PDB file
    multiple_chain: bool
        If is true, the protein file save all chains
    chain_to_work: str
        Id to save from PDB file
    remove_metals: bool
        If is true, metals from PDB file will be delete
    
    Return
    ------
    ligand_file: mol2 file
        File with ligand in mol2 format
    protein_file: mol2 file
        File with protein in mol2 format

    """
    prot = Protein.from_file(protein)
    prot.remove_all_waters()

    ### Split and select one unity if the protein is a homodimer
    ### Select only one chain
    if multiple_chain == False:
        if len(prot.chains) >= 2:
            chain = chain_to_work
            bad_chain = [c for c in [val.identifier for val in prot.chains] if c not in chain]
            for id in bad_chain:
                prot.remove_chain(id)

            for lig in prot.ligands:
                if lig.identifier.split(':')[0] in bad_chain:
                    prot.remove_ligand(lig.identifier)
                    
            for cofactor in prot.cofactors:
                if cofactor.identifier.split(':')[0] in bad_chain:
                    prot.remove_cofactor(cofactor.identifier)        

    ### Save the principal ligand
    if save_ligand == True:
        for lig in prot.ligands:
            if lig.identifier.split(':')[1] == ligand_id:
                with MoleculeWriter(path+f"/Ligand_{target_name}.mol2") as mol_writer:
                    mol_writer.write(lig)   
   
    ### Remove the ligands, metals and add hydrogens
    for l in prot.ligands:    
        prot.remove_ligand(l.identifier)

    ### Remove the ligands, metals and add hydrogens
    for l in prot.ligands:    
        prot.remove_ligand(l.identifier)

    if remove_metals == True:
        for metal in prot.metals:
            prot.remove_metal(metal)

    prot.add_hydrogens()

    ### save the protein
    with MoleculeWriter(path+f"/{target_name}_prep.mol2") as proteinWriter:
        proteinWriter.write(prot)