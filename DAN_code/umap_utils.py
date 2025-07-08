import numpy as np
import umap
import DAN_code.functions as func
from functools import partial

import pandas as pd
import datashader as ds
import datashader.transfer_functions as trans

import matplotlib.pyplot as plt
from datashader.mpl_ext import dsshow, alpha_colormap
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerLine2D
import seaborn as sns

qualitative = ListedColormap(sns.color_palette("husl", 10).as_hex())

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def load_contents(filename):
    '''
    Load the contents of a file that was saved using np.save in append mode. 
    We use this function to load the memories and weights of the DAN model saved using
    DAN_code.Callbacks.WeightEvolution at the beginning of training.

    Parameters
    ----------
    filename : str
        The path to the file to load.
    Returns
    -------
    contents : np.ndarray
        The contents of the file as a numpy array. If the file is empty or does not
        contain any data, returns None. If the file contains more than 200 arrays,
        only the first 200 are loaded.
    '''
    contents = None
    
    with open(filename, "rb") as f:
        n = 0
        while True and (n < 200):
            n += 1
            try:
                if contents is None:
                    contents = np.load(f)
                else:
                    contents = np.concatenate([contents, np.load(f)], axis = 0)
            except ValueError:
                break
    
    return contents

def calculate_overlaps(x_test, beta, name, file_suffix):
    '''
    Calculate the overlaps of test data x_test and saved DAN memories. The parameters
    other than x_test are used to find the name of the file to load.

    Parameters
    ----------
    x_test : np.ndarray
        The test data.
    beta : float
        The inverse temperature of the DAN.
    name : str
        The name of the model whose memories to load.
    file_suffix : str
        File suffix to distinguish between different saved models. Use "without_splitting"
        to load weights of a model trained without splitting steepest descent and
        "with_splitting" to load weights of a model trained with splitting steepest descent.
    Returns
    -------
    overlaps : np.ndarray
        The overlaps of the test data with the saved DAN memories.
    '''
    w = load_contents("./Data/Weights/%s_w_with_beta=%s_and_%s.npy" % (name, str(beta), file_suffix))
    
    overlaps = func.dense_cor(x_test, w.T).T
    
    return overlaps

def train_umap(overlaps, seed):
    '''
    Train a UMAP model on overlap data calculated with calculate_overlaps.

    Parameters
    ----------
    overlaps : np.ndarray
        Overlap data to train the UMAP model.
    seed : int
        Seed of random number generation for reproducibility.
    Returns
    -------
    umap_model : umap.UMAP
        The trained UMAP model.
    '''
    reducer = umap.UMAP(n_components = 2, n_neighbors = 1000, verbose = True, low_memory = False, random_state = seed)
    umap_model = reducer.fit(overlaps)
    
    return umap_model

def umap_embedding(overlaps, umap_model, beta, name, file_suffix):
    '''
    Calculate the UMAP embedding of overlap data calculated with calculate_overlaps
    and save it to a file. The parameters other than overlaps and umap_model are used
    to find the name of the file to save the embedding to.

    Parameters
    ----------
    overlaps : np.ndarray
        Overlap data to calculate the UMAP embedding.
    umap_model : umap.UMAP
        A trained UMAP model.
    beta : float
        The inverse temperature of the DAN for which the overlaps were calculated.
    name : str
        The name of the model for which the overlaps were calculated.
    file_suffix : str
        File suffix to distinguish between different models to save. Use "without_splitting"
        to save the embedding of a model trained without splitting steepest descent and
        "with_splitting" to save the embedding of a model trained with splitting steepest
        descent.
    '''
    embedding = umap_model.transform(overlaps)
    
    with open("./Data/Overlaps/%s_embedded_overlaps_with_beta=%s_and_%s.npy" % (name, str(beta), file_suffix), "wb") as f:
        np.save(f, embedding)

### May need a different environment to make it work!
def plot_umap(beta, name):
    '''
    Plot the UMAP embedding of overlap data loaded from files. The parameters
    beta and name are used to find the files to load.

    Parameters
    ----------
    beta : float
        The inverse temperature of the DAN for which the overlaps were calculated.
    name : str
        The name of the model for which the overlaps were calculated.
    '''
    fig, axes = plt.subplots(nrows = 1, ncols = 2, sharex = True, sharey = True, figsize = (10, 20))
    
    set_ylabel = True
    for file_suffix, axis in zip(["without_splitting", "with_splitting"], axes):
        with open("./Data/Overlaps/%s_embedded_overlaps_with_beta=%s_and_%s.npy" % (name, str(beta), file_suffix), "rb") as f:
            embedding = np.load(f)
        
        g = load_contents("./Data/Weights/%s_g_with_beta=%s_and_%s.npy" % (name, str(beta), file_suffix))
        
        df = pd.DataFrame(data = np.concatenate([embedding, g[np.newaxis].T], axis = -1), columns = ("x", "y", "class"))
        df["class"] = df["class"].astype("int").astype("category")
        
        artist = dsshow(df, ds.Point("x", "y"), ds.count_cat('class'), ax = axis)
    
    fig.legend(loc = 7, handles = artist.get_legend_elements(),
               framealpha = 1, edgecolor = "inherit", title = "Class")
        
    plt.show()