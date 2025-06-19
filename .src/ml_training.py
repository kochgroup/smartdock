from pycaret.classification import *
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import ADASYN
from tqdm import tqdm
import os
import pandas as pd

## Useful functions
def models_pycaret(train_set, path):
    """
    Create and get metrics for the best 5 Machine Learning models

    Parameters
    ----------
    train_set: pandas dataframe
        Values fron train the machine learning models
    test_set: pandas dataframe
        Values for test the machine leraning models
    path: dir
        Directory to save models

    Return
    ------
    metrics: pandas dataframe
        Table with 7 different metrics (Accuracy, AUC, Recall, Precision, F1, Kappa coeficient, MCC, Balanced acuraccy) 
        for evaluate ML models 
    """
    ### Charge info to process
    train = pd.read_csv(train_set)
    train = train.drop(["id", "score", "smiles"], axis=1)

    best_models = setup(data = train, target = "activity", session_id = 125, log_experiment = False,
                    normalize = True, fold_shuffle=True, use_gpu=False, fix_imbalance=True, fix_imbalance_method='ADASYN', 
                    memory=False, n_jobs=-1 #html=False, verbose=False
                    )
                    
    ### Create the best models, tune and finalize them
    models = compare_models(
        sort = "f1", include=[
            "rf", 
            "xgboost", 
            "svm", 
            "mlp"
        ],
        n_select=4
    )
    info = pull()
    tuned_models = [tune_model(model, optimize= "f1") for model in models]
    fin_models = [finalize_model(model) for model in tuned_models]

    ### Save the models
    names = info.index.tolist()
    for model, id in zip(fin_models, names):
        save_model(model, f"{path}/{id}")   

def padif_train(target, splitter_types, path_to_work):
    
    model_dir = os.path.join(path_to_work, 'models')
    os.makedirs(model_dir, exist_ok=True)

    for type in splitter_types:
        folder = f'{path_to_work}/data_to_model/{type}'
        folder2 = os.path.join(model_dir, type)
        os.makedirs(folder2, exist_ok=True)
        print(
        f'''    
        ********************************************************************************
        
        Starting to work with {target}, splitter {type}
        
        ********************************************************************************
        '''
        )  
        models_pycaret(f'{folder}/Train.csv', folder2)

