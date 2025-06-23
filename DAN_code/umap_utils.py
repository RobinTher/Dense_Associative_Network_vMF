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
    w = load_contents("./Data/Weights/%s_w_with_beta=%s_and_%s.npy" % (name, str(beta), file_suffix))
    
    overlaps = func.dense_cor(x_test, w.T).T
    #overlaps = np.linalg.lstsq(w.T, x_test.T)[0]
    #overlaps = np.linalg.lstsq(x_test.T, w.T)[0].T
    #overlaps = w
    
    return overlaps

def train_umap(overlaps, seed):
    #reducer = umap.UMAP(n_components = 2, n_neighbors = 1200, verbose = True, low_memory = False, random_state = seed)
    reducer = umap.UMAP(n_components = 2, n_neighbors = 1000, verbose = True, low_memory = False, random_state = seed)
    #reducer = umap.UMAP(n_components = 2, n_neighbors = 800, verbose = True, low_memory = False)
    #reducer = umap.UMAP(n_components = 2, n_neighbors = 500, verbose = True, low_memory = False)
    #reducer = umap.UMAP()
    umap_model = reducer.fit(overlaps)
    
    return umap_model

def umap_embedding(overlaps, umap_model, beta, name, file_suffix):
    embedding = umap_model.transform(overlaps)
    
    with open("./Data/Overlaps/%s_embedded_overlaps_with_beta=%s_and_%s.npy" % (name, str(beta), file_suffix), "wb") as f:
        np.save(f, embedding)

### May need a different environment to make it work!
def plot_umap(beta, name):
    
    fig, axes = plt.subplots(nrows = 1, ncols = 2, sharex = True, sharey = True, figsize = (10, 20))
    
    set_ylabel = True
    for file_suffix, axis in zip(["without_splitting", "with_splitting"], axes):
        with open("./Data/Overlaps/%s_embedded_overlaps_with_beta=%s_and_%s.npy" % (name, str(beta), file_suffix), "rb") as f:
            embedding = np.load(f)
        
        g = load_contents("./Data/Weights/%s_g_with_beta=%s_and_%s.npy" % (name, str(beta), file_suffix))
        
        df = pd.DataFrame(data = np.concatenate([embedding, g[np.newaxis].T], axis = -1), columns = ("x", "y", "class"))
        df["class"] = df["class"].astype("int").astype("category")
        
        artist = dsshow(df, ds.Point("x", "y"), ds.count_cat('class'), ax = axis)
        
        #axis.set_xlabel(r"$\tilde{m}^1$")
        #if set_ylabel:
        #    axis.set_ylabel(r"$\tilde{m}^2$")
        #
        #set_ylabel = False
    
    fig.legend(loc = 7, handles = artist.get_legend_elements(),
               framealpha = 1, edgecolor = "inherit", title = "Class")
        
    plt.show()
    
#def plot_umap(beta, name, file_suffix):
#    with open("./Data/Overlaps/%s_embedded_overlaps_with_beta=%s_and_%s.npy" % (name, str(beta), file_suffix), "rb") as f:
#        embedding = np.load(f)
#    
#    g = load_contents("./Data/Weights/%s_g_with_beta=%s_and_%s.npy" % (name, str(beta), file_suffix))
#    
#    plt.figure(figsize = (16, 16))
#    plt.scatter(*embedding.T, marker = ".", c = g, s = 0.5, cmap = qualitative)
#    
#    number_classes = np.max(g) + 1
#    cbar = plt.colorbar(ticks = np.linspace((1 - 1/number_classes)/2, number_classes - 1 - (1 - 1/number_classes)/2,
#                                            num = number_classes, endpoint = True))
#    cbar.ax.set_yticklabels(np.arange(number_classes))
#    # cbar.ax.hlines
#    cbar.ax.hlines(np.linspace(0, number_classes - 1, num = number_classes + 1, endpoint = True),
#                   0, 1, color = "black", linewidths = 1)
#    
#    plt.show()