#%%
from glob import glob
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

def padif_generator(file, cavity_atoms_file, docking_sln=1, type='active'):
    """
    Extract chemplp for multiple docking solutions files

    Parameters
    ----------
    file: str
        Names of docking solution files for processing
    docking_sln: int
        number of docking solution

    Return
    ------
    df: pandas dataframe
        Dataframe with all interactions
    """
    ### Open solution file and extract protein contributions and score
    with open(file, 'r') as nfile:
        f = nfile.read()
    protein_plp = f.split('> <Gold.PLP.Protein.Score.Contributions>')[docking_sln].split('$$$$')[0].strip().split('\n')
    plp_dataframe = pd.DataFrame([line.split() for line in protein_plp])
    plp_dataframe.columns = plp_dataframe.iloc[0].tolist()
        
    ### Delete and change type of some rows and columns
    plp_dataframe = plp_dataframe[plp_dataframe["AtomID"] != "AtomID"].astype('float64')
    plp_dataframe = plp_dataframe.astype({'AtomID':int})
    plp_dataframe = plp_dataframe.drop('PLP.total', axis=1)

    ### Select only atoms in cavity atoms file
    with open(cavity_atoms_file, 'r') as nfile:
        cav_f = nfile.read()
    atoms_list = [int(l) for subl in [val.split() for val in  cav_f.split('\n')] for l in subl]
    plp_dataframe = plp_dataframe[plp_dataframe["AtomID"].isin(atoms_list)]

    ### change the value for the fisrt 3 columns
    plp_dataframe["ChemScore_PLP.Hbond"] = [val*-1 if val > 0 else val for val in plp_dataframe["ChemScore_PLP.Hbond"]]
    plp_dataframe["ChemScore_PLP.CHO"] = [val*-1 if val > 0 else val for val in plp_dataframe["ChemScore_PLP.Hbond"]]
    plp_dataframe["ChemScore_PLP.Metal"] = [val*-1 if val > 0 else val for val in plp_dataframe["ChemScore_PLP.Hbond"]]

    ### Create a interaction dataframe
    atomid_values = plp_dataframe.AtomID.tolist() 
    new_column_names = [f"{col}_{atomid}" for atomid in atomid_values for col in plp_dataframe.columns if col != 'AtomID']
    interaction = plp_dataframe.drop(columns=['AtomID'])
    interaction_list = [val for sublist in interaction.values.tolist() for val in sublist]
    interaction_df = pd.DataFrame([interaction_list], columns=new_column_names)

    ### Add score and id for each 
    interaction_df['score'] = round(float(f.split('> <Gold.PLP.Fitness>')[1].split('> <Gold.PLP.PLP>')[0].strip()),3)

    interaction_df['id'] = file.split('/')[-1].replace("_sln.sdf", "") + "_" + str(docking_sln)

    return interaction_df

def padif_dataframe(path_solutions, padif_type='active', padif_ref=None, path=None):
    padif_final = []
    for file in tqdm(glob(f'{path_solutions}/*.sdf')):
        padifs_sln = [
            padif_generator(
                file,
                f'{path}/cavity.atoms',
                num+1
            ) for num in range(10)
        ]
        padif_df = pd.concat((df.T for df in padifs_sln), axis=1).T
        padif_df = padif_df.fillna(0.0)      
        if padif_type == 'active':
            padif_df = padif_df[padif_df.id.str.endswith('1')].reset_index(drop=True)
            padif_df['activity'] = 1
            padif_final.append(padif_df)
        else:
            ### Calculate the distance of actives from test and keep only the most un-similar padif
            ref = padif_ref.drop(['id', 'score', 'activity'], axis=1)
            padif_df_copy = padif_df.copy()
            padif_df_copy = padif_df_copy.drop(['id', 'score'], axis=1)            
            list_add = [col for col in ref.columns.tolist() if col not in padif_df_copy.columns.tolist()]
            list_add_2 = [col for col in padif_df_copy.columns.tolist() if col not in ref.columns.tolist()]
            padif_df_copy = padif_df_copy.reindex(columns=[*padif_df_copy.columns.tolist(), *list_add], fill_value=0.0)
            ref = ref.reindex(columns=[*ref.columns.tolist(), *list_add_2], fill_value=0.0)            
            padif_am = ref.sum(axis=0).values
            padif_df['cos_dis'] = cosine_distances([padif_am], padif_df_copy)[0] 
            padif_df = padif_df[padif_df.cos_dis == padif_df.cos_dis.max()]
            if padif_type == 'decoys':
                padif_df['activity'] = 0
                padif_final.append(padif_df)
            elif padif_type == 'test':
                padif_final.append(padif_df)
    
    padif_final = pd.concat(padif_final)
    padif_final = padif_final.fillna(0.0)
    
    if padif_type == 'test':
        to_move = ['score', 'id']
    else:
        to_move = ['score', 'id', 'activity']
        
    order_1 =[col for col in padif_df.columns if col not in to_move] + to_move
    padif_final = padif_final[order_1]

    return padif_final

