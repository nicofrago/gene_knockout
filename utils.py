import umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
   accuracy_score, 
   precision_score, 
   f1_score, 
   recall_score,
   confusion_matrix
)

def get_genes_intersection(genesa, genesb):
    """
    Computes the intersection of two lists of genes.

    Parameters
    ----------
    genesa : list
        The first list of genes.
    genesb : list
        The second list of genes.

    Returns
    -------
    intersection_genes : list
        The intersection of the two lists of genes.

    """
    
    intersection_genes = list(set(genesa).intersection(set(genesb)))
    print('genes a:', len(genesa), 'genes b:', len(genesb), 'intersection:', len(intersection_genes))
    return intersection_genes

def plot_umaps(embeddings:list, adata, limit=None, granularity=1, titles:list=[]):
    """
    Plot the UMAP representation of the embeddings.

    Parameters
    ----------
    embeddings : list
        A list of embeddings to be plotted.
    adata : AnnData
        The AnnData object containing the cell type labels.
    limit : int or None
        The number of cells to be plotted. If None, all cells will be plotted.
    granularity : int
        The granularity of the cell type labels to be used. 1 for LVL1, 2 for LVL2, 3 for LVL3.
    titles : list
        A list of titles for the subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots.
    """
    if len(titles) == 0:
        titles = ['UMAP of Reference Data'] * len(embeddings)
    granularities = {1: 'LVL1', 2: 'LVL2', 3: 'LVL3'}
    granularity = granularities[granularity]
    labels = adata.obs[granularity][:limit]
    fig, ax = plt.subplots(1, len(embeddings), figsize=(10*len(embeddings), 10))
    for i, embedding in enumerate(embeddings):
        embedding = embedding[:limit]
        # Create UMAP reducer and fit the embeddings
        reducer = umap.UMAP(min_dist=0.1, n_components=2, n_epochs=None, n_neighbors=4)
        mapper = reducer.fit(embedding)

        # Create a DataFrame for plotting
        plot_df = pd.DataFrame(mapper.embedding_, columns=['px', 'py'])
        plot_df['Cell Type'] = labels.values    
        # Plot the UMAP results
        sns.set_style('white')
        sns.scatterplot(data=plot_df, x='px', y='py', hue='Cell Type', sizes=(50, 200), ax=ax[i], palette="pastel")
        ax[i].set_title(titles[i])
    return fig

def get_evaluations(name_data_set, y_true, y_pred) -> dict:
    """
    Evaluate the performance of a model on a given dataset.

    Parameters
    ----------
    name_data_set : str
        The name of the dataset to be evaluated.
    y_true : array-like
        The true labels of the dataset.
    y_pred : array-like
        The predicted labels of the dataset.

    Returns
    -------
    evaluation : dict
        A dictionary containing the accuracy, precision, f1-score, and recall of the model on the given dataset.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    print(f"{name_data_set} accuracy: {(accuracy*100):.1f}%")
    print(f"{name_data_set} precision: {(precision*100):.1f}%")
    print(f"{name_data_set} f1: {(f1*100):.1f}%")
    print(f"{name_data_set} recall: {(recall*100):.1f}%")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
        "recall": recall,
    }

def plot_costs_matrix(y_true, y_preds:list, classes:list, titles = ['Original', 'Knockout']):
    """
    Plots confusion matrices for original and knockout predictions.

    Parameters
    ----------
    y_true : list or array-like
        The true class labels.
    y_preds : list
        A list containing arrays of predicted class labels.
    classes : list
        The list of class names corresponding to the labels.

    Returns
    -------
    None
    Displays a plot containing two confusion matrices: one for the original predictions and one for the knockout predictions.
    """    
    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    for i, y_pred in enumerate(y_preds):
        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = pd.DataFrame(cm, index=classes[:cm.shape[0]], columns=classes[:cm.shape[1]])
        sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues", ax=ax[i])
        ax[i].set_title(titles[i])
    plt.show()


def plot_training(
        train_loss:list,
        val_loss:list,
        epochs:int,
        save_in:str
):
    """
    Plots training and validation loss during training

    Args:
        train_loss (list): List of training losses
        val_loss (list): List of validation losses
        epochs (int): Number of epochs
        save_in (str): Directory to save the plots
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_loss, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_in)
    plt.show()
