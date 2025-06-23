import numpy as np
import umap
import DAN_code.functions as func

### May need a different environment to make it work!
def plot_umap(x_test, beta, name, file_suffix):
    x_test = x_test[400:2300:5]
    w = np.array([])
    g = np.array([])
    
    with open("./Data/Weights/" + name + "_w_with_beta=" + str(beta)
              + "_and_%s.npy" % file_suffix, "rb") as f:
        while True:
            try:
                w = np.concatenate([w, np.load(f)], axis = 0)
            except OSError:
                break
    
    with open("./Data/Weights/" + name + "_g_with_beta=" + str(beta)
              + "_and_%s.npy" % file_suffix, "rb") as f:
        while True:
            try:
                g = np.concatenate([g, np.load(f)], axis = 0)
            except OSError:
                break
    
    w = w.T
    #w = w.reshape(-1, x_test.shape[-1]).T
    #g = g.flatten()
    overlaps = func.dense_cor(x_test - np.mean(x_test, axis = 1, keepdims = True), w)

    reducer = umap.UMAP(n_components = 2, n_neighbors = 1000, verbose = True, low_memory = False)
    embedding = reducer.fit_transform(overlaps)
    
    plt.scatter(embedding[0], embedding[1])
    plt.show()
    # umap.plot.point(embedding, labels = g)