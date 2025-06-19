import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import *
from sklearn.metrics import balanced_accuracy_score
from rdkit.ML.Scoring.Scoring import CalcEnrichment


def predictions_models(splitters_list, path):
    """
    Test the models and extract the metrics (Balanced accuracy, enrichment fasctors [1%, 25%]) for models

    Parameters
    ----------
    splitters_list: str
        Values with names of splitters used for create the models
    target: str
        name of target
    path: str
        Directory with test files

    Return
    ------
    metrics: pandas dataframe
        Dataframe with metrcis calculated for each model tarined
    predictions: pandas dataframe
        Dataframe with prediction for each molecule tested
    """

    data_metrics = []
    data_predictions = []

    ### Extract all metrcis for each type of splitter
    for type in splitters_list:

        predictions = []
        ### Open test files
        test = pd.read_csv(f'{path}/data_to_model/{type}/Test.csv')

        ### Calculate metrics for scoring functions:
        ### Organize values for CHEMPLP scoring
        test_1 = test.astype({"score":"float64"})
        test_1 = test_1.sort_values(by=["score"], ascending=False)
        ### Select te top of the list as actives according to the scoring
        test_1["prediction_label"] = [1 for val in range(int(len(test_1)*0.25))] + [0 for val in range(len(test_1) - int(len(test)*0.25))]
        ### Organize the data according with docking scoring 
        act = test_1[test_1["prediction_label"] == 1].sort_values(["score"],ascending=False)
        ina = test_1[test_1["prediction_label"] == 0].sort_values(["score"],ascending=False)
        pred_cs = pd.concat([act, ina])
        pred_cs['model'] = 'CHEMPLP_scoring'
        bal_acc = [round(balanced_accuracy_score(pred_cs.activity, pred_cs.prediction_label),4)]
        ### Extract list for calculate enrichment factors 
        perfect_scores = [[1]]*len(pred_cs[pred_cs.activity == 1]) + [[0]]*len(pred_cs[pred_cs.activity == 0])
        perfect_ef = CalcEnrichment(perfect_scores, 0, [0.01, 0.25])
        scores = [[x] for x in pred_cs.activity]
        ef = CalcEnrichment(scores, 0, [0.01, 0.25])
        ef_std = [[round(x/y, 4) for x, y in zip(ef, perfect_ef)]]
        md = ['CHEMPLP_scoring']


        ### With all models extract Balanced accuracy and enrichment factors
        for file in glob.glob(f'{path}/models/{type}/*pkl'):
            ### Open models and predict
            model = load_model(file[:-4])
            pred = predict_model(model, test)
            ### Organize the data according with docking scoring 
            act = pred[pred["prediction_label"] == 1].sort_values(["score"],ascending=False)
            ina = pred[pred["prediction_label"] == 0].sort_values(["score"],ascending=False)
            pred = pd.concat([act, ina])
            pred['model'] = file.split("/")[-1].split(".")[0]
            predictions.append(pred)
            bal_acc.append(round(balanced_accuracy_score(pred.activity, pred.prediction_label),4))
            ### Extract list for calculate enrichmene factors 
            pefrfect_scores = [[1]]*len(pred[pred.activity == 1]) + [[0]]*len(pred[pred.activity == 0])
            pefrfect_ef = CalcEnrichment(pefrfect_scores, 0, [0.01, 0.25])
            scores = [[x] for x in pred.activity]
            ef = CalcEnrichment(scores, 0, [0.01, 0.25])
            ef_std.append([round(x/y, 4) for x, y in zip(ef, pefrfect_ef)])
            md.append(file.split("/")[-1].split(".")[0])
                
        ### create a dataframe for metrics
        metrics = pd.DataFrame(
            list(zip(bal_acc, md)), 
            columns=["BA", "model"]
        )
        ### Add enrichment factors for metrics and save
        enrichment = pd.DataFrame(ef_std, columns=["NEF1%", "NEF25%"]) 
        metrics = pd.concat([metrics, enrichment], axis=1)
        metrics["type"] = type   
        data_metrics.append(metrics)

        ### Create a dataframe for all test predictions
        final_pred = pd.concat(predictions).reset_index(drop=True)
        final_pred = pd.concat([final_pred, pred_cs]).reset_index(drop=True)
        final_pred['type'] = type
        data_predictions.append(final_pred)
    
    ### concat values from predictions and metrics
    metrics = pd.concat(data_metrics).reset_index(drop=True)
    metrics['class'] = ["scoring" if val == 'CHEMPLP_scoring' else 'model' for val in metrics.model]
    predictions = pd.concat(data_predictions).reset_index(drop=True)

    return metrics, predictions

def figures_from_metrics(prediction_df, metrics_df, path):

    ### Create directory for save figures
    figures_path = os.path.join(path, 'figures')
    os.makedirs(figures_path, exist_ok=True)

    ### SCORING distribution
    ### Select only the data for CHEMPLP score
    cs = prediction_df[prediction_df['model'] == 'CHEMPLP_scoring']
    cs = cs.sort_values(['activity'], ascending=True)
    ### Calculate limits for histogram plot
    hist, edges = np.histogram(cs.score, bins=10)
    max_percent = max((hist / hist.sum()) *100)
    scores_limits = [min(cs.score) - 2, max(cs.score) + 2]


    ### Create a subplot for histograms
    fig, ax = plt.subplots(1,3, figsize=(10,4))
    axes = ax.ravel()
    fig.subplots_adjust(hspace=0.5,wspace=0.4)
    sns.set(style="white")

    ### For each kind of splitter create a histogram
    for val, num in zip(cs.type.unique(), axes):
        ### Create a histogram
        g = sns.histplot(data=cs[cs["type"] == val], x="score", hue="activity", alpha=.5, fill=True, 
                        palette=['#ff7f0e', "#007fff", '#d9f0a3', '#228343'], ax=num, stat="percent",
                        legend=True, element='step', hue_order=[1,0], bins=10)
        ### set style keys
        g.set(title=val)
        sns.despine()    
        g.set_ylim(0,max_percent)
        g.set_xlim(scores_limits[0], scores_limits[1])

    plt.savefig(f'{figures_path}/histogram_scoring_distribution.png', dpi=300, transparent=True, bbox_inches="tight")

    ### See Balanced accuracy for models
    ### Select style parameters
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.set(rc={'figure.figsize': (15, 15)})
    sns.set(font_scale=2)
    ### Graph the Balanced accuracy per models
    sns.boxplot(y='type', x='BA', orient='h', hue='class', data=metrics_df)
    sns.swarmplot(y='type', x='BA', orient='h', hue='model', data=metrics_df, size=16, 
                palette=sns.color_palette('Set2',n_colors=8)[:5])
    ax.set(ylabel="splitter",xlabel="$balanced - accuracy$", xlim=(0, 1))
    ### Modify legend and borders
    _ = ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
    sns.despine(right=True, top=True)
    ### Save the Plot
    plt.savefig(f'{figures_path}/Balance_accuracy_per_model.png', dpi=300, transparent=True, bbox_inches="tight")

def test_predictions(padif_ref, padif_test, path, splitters_list):
    list_add = [col for col in padif_ref.columns.tolist() if col not in padif_test.columns.tolist()]
    test_f = padif_test.reindex(columns=[*padif_test.columns.tolist(), *list_add], fill_value=0.0)
    per_spt =[]
    for type in splitters_list:
        predictions = []
        model_n = []        
        ### With all models extract Balanced accuracy and enrichment factors
        for file in glob.glob(f'{path}/models/{type}/*pkl'):
            ### Open models and predict
            model = load_model(file[:-4])
            pred = predict_model(model, test_f)
            predictions.append(pred.prediction_label.tolist())
            model_n.append(file.split("/")[-1].split(".")[0])
        pr = pd.DataFrame(predictions).T
        pr.columns = [f'{mod}_{type}_pred-label' for mod in model_n]
        per_spt.append(pr)
    pred_spt = pd.concat(per_spt, axis=1).reset_index(drop=True)
    pred_spt['final_pred'] = pred_spt.apply(lambda row: 1 if (row == 1).all() else 0, axis=1)
    test_data = padif_test[['id', 'score']]
    final = pd.concat([test_data, pred_spt], axis=1)
    return final
