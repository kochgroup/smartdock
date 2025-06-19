import os
import pandas as pd
from deepchem import data, splits
from sklearn.model_selection import train_test_split

### Define splitness funtions

### Create a funtion for randoms splitness
def random_stratify_split(df):
    ### Select data and split
    y = df.pop("activity").to_frame()
    x = df

    X_train, X_test, y_train, y_test =train_test_split(
            x, y,stratify=y, test_size=0.1)
    data = pd.concat([X_train, y_train], axis=1)
    data_unseen = pd.concat([X_test, y_test], axis=1)
    return data, data_unseen

def molecular_stratify_split(df, molecular_method):
    x = df.loc[:, df.columns != 'activity'].values
    y = df[['activity']].astype('float')['activity'].values

    dataset = data.DiskDataset.from_numpy(
        X = x,
        y = y,
        w = x,
        ids = df.smiles.tolist()
    )
    if molecular_method == 'scaffold':
        scf_split = splits.ScaffoldSplitter()
        train, test = scf_split.train_test_split(dataset, frac_train=0.9)
    elif molecular_method == 'fingerprint':
        fp_split = splits.FingerprintSplitter()
        train, test = fp_split.train_test_split(dataset, frac_train=0.9)
    else:
        print('Incorrect molecular splitter')
    
    train_f = df[df.smiles.isin(train.ids.tolist())]
    test_f = df[df.smiles.isin(test.ids.tolist())]

    
    return train_f, test_f


import warnings
warnings.simplefilter('ignore', pd.errors.DtypeWarning)

def to_split(target, padif_folder, path_to_work, method='random'):

    if method == 'random':
        splitter = random_stratify_split
    else:
        splitter = molecular_stratify_split

    
    print(f'''
        ********************************************************************************
        Startng to process with {target}, with {method} splitter
        ********************************************************************************
    \n''')
    ### split data
    set = pd.read_csv(f'{padif_folder}/{target}_PADIF.csv')
    if method == 'scaffold':
        actives = set[set['activity'] == 1]
        decoys = set[set['activity'] == 0]
        data_act, data_unseen_act = splitter(df=actives, molecular_method = 'scaffold')
        data_dec, data_unseen_dec = splitter(df=decoys, molecular_method = 'scaffold')
        data = pd.concat([data_act, data_dec])
        data_unseen = pd.concat([data_unseen_act, data_unseen_dec])
    elif method == 'fingerprint':
        actives = set[set['activity'] == 1]
        decoys = set[set['activity'] == 0]
        data_act, data_unseen_act = splitter(df=actives, molecular_method = 'fingerprint')
        data_dec, data_unseen_dec = splitter(df=decoys, molecular_method = 'fingerprint')
        data = pd.concat([data_act, data_dec])
        data_unseen = pd.concat([data_unseen_act, data_unseen_dec])
    else:
        data, data_unseen = splitter(df=set)

    print(
    f'{method} set\n'
    f'actives proportion in training set is: {round((data.activity.value_counts()[1] / len(data)), 3)}\n'
    f'actives proportion in test set is: {round((data_unseen.activity.value_counts()[1] / len(data_unseen)), 3)}'
    )
   
    ### save dataframes
    data.to_csv(f'/{path_to_work}/Train.csv', sep=',', index=False)
    data_unseen.to_csv(f'/{path_to_work}/Test.csv', sep=',', index=False)

       
