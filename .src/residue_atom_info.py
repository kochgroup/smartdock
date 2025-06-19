#%%
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def atom_interaction_info(path):
      with open(f'{path}/plp_protein.mol2', 'r') as nfile:
            f = nfile.read()
      ### Select the atoms in columns from protein in mol2 file
      atoms = f.split('@<TRIPOS>ATOM')[1].split('@<TRIPOS>BOND')[0].strip().split('\n')
      values_list = [val.split() for val in atoms]
      ### Add "unclassified" value to sublist without type of atoms
      for sub in values_list:
            if len(sub) < 9: sub.insert(5, 'unclasssied')
      ### Change column names and types of AtomID
      protein_atoms = pd.DataFrame(values_list)
      protein_atoms.columns = ['AtomID', 'atom', 'x', 'y', 'z', 'atomType', 'residueNum', 'residueName', 'unk']
      protein_atoms = protein_atoms.astype({'AtomID':int})

      return protein_atoms

def atoms_by_activity(padif_df, protein_df):
      ### Open the padif file and select the fingerprints by type of activity
      active = padif_df[padif_df['activity'] == 1]
      decoys = padif_df[padif_df['activity'] == 0]
      ### Actives columns with interactions
      active =  active.iloc[:, :-4]
      active = active.loc[:, active.sum() > 0]
      cols_act = active.columns
      ## Decoys columns with interactions
      decoys =  decoys.iloc[:, :-4]
      decoys = decoys.loc[:, decoys.sum() > 0]
      cols_dec = decoys.columns
      ### Select types of columns
      types_df = protein_df[['AtomID', 'atomType', 'atom', 'residueNum', 'residueName']]
      ### Interaction residues for actives dockings
      interaction_act_df = pd.DataFrame([int(val.split('_')[-1]) for val in cols_act], columns=['AtomID'])
      interaction_act_df = interaction_act_df.merge(types_df, on='AtomID')
      interaction_act_df['interactionType'] = ['_'.join(val.split('_')[:2]) if val.startswith('Chem') else val.split('_')[0] for val in cols_act]
      ### Interaction residues for decoys dockings
      interaction_dec_df = pd.DataFrame([int(val.split('_')[-1]) for val in cols_dec], columns=['AtomID'])
      interaction_dec_df = interaction_dec_df.merge(types_df, on='AtomID')
      interaction_dec_df['interactionType'] = ['_'.join(val.split('_')[:2]) if val.startswith('Chem') else val.split('_')[0] for val in cols_dec]
      ###
      all_df = pd.concat([interaction_act_df, interaction_dec_df]).reset_index(drop=True)
      ### Create groups for display in chimera
      act_unique = set(interaction_act_df['AtomID']) - set(interaction_dec_df['AtomID'])
      dec_unique = set(interaction_dec_df['AtomID']) - set(interaction_act_df['AtomID'])
      all_atoms = set(interaction_act_df['AtomID']) | set(interaction_dec_df['AtomID'])
      ###
      act_df = all_df[all_df.AtomID.isin(list(act_unique))]
      dec_df = all_df[all_df.AtomID.isin(list(dec_unique))]
      all_df_f = all_df[all_df.AtomID.isin(list(all_atoms))]

      return act_df, dec_df, all_df_f

### Create function for select display atoms
def write_chimera_commands(protein, all_atoms, active_atoms, decoys_atoms, folder_to_save_picture, filename):
    with open(filename, 'w') as file:
        # Write the command to load the protein model (edit path as needed)
        file.write(f"open {protein}\n")
        
        # Convert the list of atoms to a string and format it for the select command
        atom_selection = ' '.join(map(str, all_atoms))
        file.write(f"sel :{atom_selection}\n")
        
        # Write the command to apply balls and sticks representation
        file.write("repr bs sel\n")
        
        # Write the command to apply balls and sticks representation
        file.write("sel up\n")
        
        # Display atoms
        file.write("disp sel\n")
        
        # Hide non-polar Hydrogen atoms
        file.write("~disp HC\n")
                
        # Invert selection 
        file.write("sel invert\n")
        
        # Change the color to the rest of the protein
        file.write("color gray\n")
        
        # Add transparency to this part
        file.write("tran 50\n")
        
        # Convert the list of atoms to a string and format it for the select command
        atom_selection = ' '.join(map(str, all_atoms))
        file.write(f"sel :{atom_selection}\n")
        
        # Change color to interaction atoms to orange
        file.write("color orange sel\n")

        # Convert the list of atoms to a string and format it for the acitve atoms
        atom_selection = ' '.join(map(str, active_atoms))
        file.write(f"sel :{atom_selection}\n")

        # Change color to interaction atoms to orange
        file.write("color olive drab sel\n")

        # Convert the list of atoms to a string and format it for the acitve atoms
        atom_selection = ' '.join(map(str, decoys_atoms))
        file.write(f"sel :{atom_selection}\n")

        # Change color to interaction atoms to orange
        file.write("color brown sel\n")

        # Convert the list of atoms to a string and format it for the select command
        atom_selection = ' '.join(map(str, all_atoms))
        file.write(f"sel :{atom_selection}\n")
        
        # Write the command to focus view on selected atoms
        file.write("focus sel\n")

        # Write the command to unselect the atoms
        file.write(f"~sel :{atom_selection}\n")
        
        # Write the command to activate presets publication 1
        file.write("preset apply pub 1\n")
        
        # Write the command to set the background transparent
        file.write("set bgTransparency\n")
        
        # Write the command to vae picture of this protein
        file.write(f"copy file {folder_to_save_picture}/PADIF_atoms.png dpi 300\n")

        # Write the command to activate presets publication 1
        file.write("preset apply pub 1\n")  

def inter_by_mols(val, act, dec):
      if val in list(act):
            return 'Only in actives'
      elif val in list(dec):
            return 'Only in decoys'
      else:
           return 'Shared'
      
def residue_interactions(padif, path_figure, act_ids, dec_ids):
      padif_p =  padif.iloc[:, :-4]
      padif_p = padif_p.loc[:, padif_p.sum() > 0]
      cols_padif = padif_p.columns

      ### Select types of columns
      types_df = protein_df[['AtomID', 'atomType', 'atom', 'residueName']]

      padif_interactions = pd.DataFrame([int(val.split('_')[-1]) for val in cols_padif], columns=['AtomID'])
      padif_interactions = padif_interactions.merge(types_df, on='AtomID')
      padif_interactions['interactionType'] = ['_'.join(val.split('_')[:2]) if val.startswith('Chem') else val.split('_')[0] for val in cols_padif]

      padif_interactions['inte_by_mols'] = padif_interactions.AtomID.apply(lambda id: inter_by_mols(id, act_ids, dec_ids))
      
      cols = ['interactionType', 'atomType', 'inte_by_mols']
      title = ['Type of interaction', 'Type of atom', 'Interaction by molecules']
      fig, ax = plt.subplots(1,3,figsize=(16,10))
      for num, col in enumerate(zip(cols, title)):
            pivot_table = padif_interactions.pivot_table(index='residueName', columns=col[0], aggfunc='size', fill_value=0.0)
            fig = sns.heatmap(pivot_table, annot=True, cmap='YlGn',ax=ax[num], cbar=False)
            fig.set_ylabel('')
            fig.set_xlabel('')
            fig.set_title(col[1],fontdict={'fontsize':16})

      plt.savefig(f'{path_figure}/residue_interacrions.png', dpi=300, bbox_inches='tight')

      return padif_interactions      


