import numpy as np
import tensorflow as tf

import DAN_code.functions as func
import DAN_code.normalization as norm

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.animation import ArtistAnimation
import cmasher as cmr
import seaborn as sns

import string
from datetime import datetime
uppercase_array = np.array(list(string.ascii_uppercase), dtype = "str")

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def truncate_colormap(cmap, minval = 0, maxval = 1, n = 256):
    
    new_cmap = LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n = cmap.name, a = minval, b = maxval),
        cmap(np.linspace(minval, maxval, n, endpoint = True)))
    
    return new_cmap

# Set red -> white -> blue and white -> blue colormaps used for all the plotting
coldhot = cmr.fusion
cold = truncate_colormap(cmr.fusion, 0.5, 1, 128)
nice_blue = cmr.fusion(0.75)

qualitative = ListedColormap(sns.color_palette("husl", 10).as_hex())

### Return predicted hard labels of x (predicted classes) and corresponding soft labels (confidence values).
def classes_and_confidences(x, model):
    f = model.predict(x)
    p_hard = np.argmax(f, axis = 1)
    p_soft = np.around(np.max(func.softmax(f, axis = 1), axis = 1), decimals = 2)
    
    return p_hard, p_soft

### Show image data or first layer weights.
def plot_images(images, dimensions = None, animated = False, title = None, labeled = True, filename = None, fignum = None):
    if images.ndim == 1:
        images = images[np.newaxis]
    
    # Use animated = False to use the plotted weights in an animation.
    if animated and (fignum is None):
        # The image is put onto the current figure if animated.
        fignum = 0
    # else:
        # A new figure is created if not animated.
        # fignum = None
    
    sqrt_size = int(images.shape[-1]**(1/2))
    if images.ndim == 2 and images.shape[-1] == sqrt_size**2:
        image_height = sqrt_size
        image_width = sqrt_size
    elif images.ndim == 3 or images.ndim == 4:
        image_height = images.shape[-2]
        image_width = images.shape[-1]
        images = images.reshape(-1, image_height, image_width)
    else:
        raise ValueError("Image shape not supported.")
    
    # Infer image dimensions if not provided
    (number_images, *image_shape) = images.shape
    if dimensions == None:
        height = int(np.ceil(number_images**(1/2)))
        width = height
    else:
        height = dimensions[0]
        width = dimensions[1]
    
    # Adjust images to the height and width provided
    area = height * width
    if number_images < area:
        images = np.concatenate([images, np.zeros((area - number_images, *image_shape))])
    elif number_images > area:
        images = images[: area]
    else:
        pass
    
    try:
        cap = norm.array_max_norm(images, keepdims = False)
        images = images.reshape(height, width, image_height, image_width).swapaxes(1, 2)
        images = images.reshape(image_height * height, image_width * width)
    # Plot an empty image if images is empty
    except ValueError:
        cap = 1
        images = np.zeros((image_height, image_width))
    
    image = plt.matshow(images, fignum = fignum, cmap = coldhot, vmin = -cap, vmax = cap,
                        animated = animated)
    if labeled:
        plt.tick_params(bottom = False, right = False, labelbottom = False, labelright = False)
        
        plt.xticks(ticks = np.arange(0.5 * image_width - 0.5, image_width * (width + 0.5) - 0.5, image_width),
                   labels = uppercase_array[np.arange(width).astype("int") % 26])
        
        plt.yticks(ticks = np.arange(0.5 * image_height - 0.5, image_height * (height + 0.5) - 0.5, image_height),
                   labels = uppercase_array[np.arange(height).astype("int") % 26])
    else:
        plt.tick_params(top = False, bottom = False, left = False, right = False,
                        labeltop = False, labelbottom = False,  labelleft = False, labelright = False)
    
    plt.vlines(np.arange(image_width - 0.5, image_width * width - 0.5, image_width),
               -0.5, image_height * height - 0.5, color = "black",
               linestyles = "dashed", linewidths = 1)
    
    plt.hlines(np.arange(image_height - 0.5, image_height * height - 0.5, image_height),
               -0.5, image_width * width - 0.5, color = "black",
               linestyles = "dashed", linewidths = 1)
    
    if animated:
        return image
    else:
        # Add an optional title
        if title is not None:
            plt.title(title)
        
        if labeled:
            plt.colorbar(ticks = [-cap, 0, cap], shrink = 0.8, aspect = 20*0.8)
        
        if filename is not None:
            plt.savefig("./Data/Figures/%s.png" % filename)
        
        if fignum is None:
            plt.show()
        
        return height, width

### Plot label-like data or second layer weights.
def plot_labels(predictions, labels, dimensions = None, animated = False, labeled = True, filename = None, fignum = None):
    # Use animated = False to use the plotted weights in an animation.
    if animated & (fignum is None):
        # The image is put onto the current figure if animated.
        fignum = 0
    # else:
        # A new figure is created if not animated.
        # fignum = None
    
    # Infer image dimensions if not provided
    number_images = predictions.shape[0]
    if dimensions == None:
        height = int(np.ceil(number_images**(1/2)))
        width = height
    else:
        height = dimensions[0]
        width = dimensions[1]
    
    # if model.loss.supervised and not animated:
        # pass
        # predictions = classes_and_confidences(images, model)
    
    cap = norm.array_max_norm(labels, keepdims = False)
    
    image = plt.matshow(labels, fignum = fignum, cmap = cold, vmin = 0, vmax = cap, animated = animated)
    plt.vlines(np.arange(0.5, labels.shape[1] - 0.5), -0.5, labels.shape[0] - 0.5, color = "black",
               linestyles = "dashed", linewidths = 1)
    
    if labeled:
        plt.xlabel("Memory")
        plt.ylabel("Class")
        
        tick_locations, tick_labels = plt.yticks(np.arange(labels.shape[0] - 1))
        
        height_labels = np.arange(height).astype("int")
        width_labels = np.arange(width).astype("int")
        
        height_labels = uppercase_array[height_labels % 26]
        width_labels = uppercase_array[width_labels % 26]
        
        memory_indices = np.array(np.meshgrid(height_labels, width_labels)).reshape(2, -1)
        memory_indices = np.char.add(memory_indices[1], memory_indices[0])
        
        plt.xticks(np.arange(labels.shape[1])[: number_images], labels = memory_indices[: number_images], rotation = 45)
        
        main_axis = plt.gca()
        
        main_axis.xaxis.tick_bottom()
        
        secondary_axis = main_axis.secondary_xaxis("top")
        
        secondary_axis.set_xticks(np.arange(labels.shape[1])[: number_images])
        secondary_axis.set_xticklabels(predictions.astype("int8"))
    else:
        plt.tick_params(top = False, bottom = False, left = False, right = False,
                        labeltop = False, labelbottom = False,  labelleft = False, labelright = False)
    
    # if model.loss.supervised and not animated:
        # secondary_axis.set_xticklabels(predictions[0].astype("int8"))
    # else:
        # secondary_axis.set_xticklabels([])
    
    if animated:
        return image
    else:
        if labeled:
            plt.colorbar(ticks = [0, cap], shrink = 0.8, aspect = 20*0.8)
        
        if filename is not None:
            plt.savefig("./Data/Figures/%s.png" % filename)
        
        if fignum is None:
            plt.show()

def plot_activations(activations, w = None, g = None, dimensions = None, title = None):
    # Infer image dimensions if not provided
    number_images = activations.shape[0]
    if dimensions == None:
        height = int(np.ceil(number_images**(1/2)))
        width = height
    else:
        height = dimensions[0]
        width = dimensions[1]
    
    container = plt.bar(np.arange(number_images), height = activations, width = 0.8, facecolor = nice_blue,
                        edgecolor = "black", linestyle = "dashed", linewidth = 1)
    
    if g is not None:
        classes = np.argmax(g, axis = 0)
        plt.bar_label(container, classes)
    
    plt.xlabel("Memory")
    plt.ylabel("Activation")
    
    plt.xlim(-0.5, number_images - 0.5)
    bottom, top = plt.ylim()
    # plt.ylim(bottom, np.maximum(top, 3 * np.mean(activations)))
    plt.ylim(bottom, top + np.maximum(activations[np.floor(0.4 * number_images).astype("int") - 1] + 0.05 - 0.4 * top, 0))
    
    height_labels = np.arange(height).astype("int")
    width_labels = np.arange(width).astype("int")
    
    height_labels = uppercase_array[height_labels % 26]
    width_labels = uppercase_array[width_labels % 26]
    
    memory_indices = np.array(np.meshgrid(height_labels, width_labels)).reshape(2, -1)
    memory_indices = np.char.add(memory_indices[1], memory_indices[0])
    
    plt.xticks(np.arange(activations.shape[0])[: number_images], labels = memory_indices[: number_images], rotation = 45)
    
    if title is not None:
        plt.title(title)
    
    if w is not None:
        base_fig = plt.gcf()
        # axis = base_fig.add_subplot(3, 3, 5)
        left, bottom, width, height = [0.425, 0.4, 0.4, 0.4]
        axis = base_fig.add_axes([left, bottom, width, height])
        plot_images(w, fignum = 0)
    
    plt.show()

def plot_linear_decomposition(x_true, x_init, w_mean_given_target, w_mean, x_res, x_init_coef, w_mean_given_target_coef, w_mean_coef, x_res_coef, dimensions = None, title = None):
    
    base_fig = plt.figure(figsize = (6*6.4, 4.8))
    
    left, bottom, width, height = [0, 0, 1/6, 1]
    axis = base_fig.add_axes([left, bottom, width, height])
    plot_images(x_true, dimensions, labeled = False, fignum = 0)
    
    x_init_coef = np.format_float_positional(x_init_coef, precision = 3, unique = False, sign = False)
    plt.annotate(r"$=\!%s\ldots\!\times$" % x_init_coef, (1/6 - 1/24 + 1/168, 0.5), xycoords = "figure fraction", fontsize = 36)
    left, bottom, width, height = [1/6 + 1/24, 0, 1/6, 1]
    axis = base_fig.add_axes([left, bottom, width, height])
    plot_images(x_init, dimensions, labeled = False, fignum = 0)
    
    w_mean_given_target_coef = np.format_float_positional(w_mean_given_target_coef, precision = 3, unique = False, sign = True)
    plt.annotate(r"$%s\ldots\!\times$" % w_mean_given_target_coef, (2/6 + 1/112, 0.5), xycoords = "figure fraction", fontsize = 36)
    left, bottom, width, height = [2/6 + 1/12, 0, 1/6, 1]
    axis = base_fig.add_axes([left, bottom, width, height])
    plot_images(w_mean_given_target, dimensions, labeled = False, fignum = 0)
    
    w_mean_coef = np.format_float_positional(-w_mean_coef/2, precision = 3, unique = False, sign = True)
    plt.annotate(r"$%s\ldots\!\times$" % w_mean_coef, (3/6 + 1/24 + 1/112, 0.5), xycoords = "figure fraction", fontsize = 36)
    left, bottom, width, height = [3/6 + 3/24, 0, 1/6, 1]
    axis = base_fig.add_axes([left, bottom, width, height])
    plot_images(w_mean, dimensions, labeled = False, fignum = 0)
    
    x_res_coef = np.format_float_positional(x_res_coef, precision = 3, unique = False, sign = True)
    plt.annotate(r"$%s\ldots\!\times$" % x_res_coef, (4/6 + 1/12 + 1/112, 0.5), xycoords = "figure fraction", fontsize = 36)
    left, bottom, width, height = [5/6, 0, 1/6, 1]
    axis = base_fig.add_axes([left, bottom, width, height])
    plot_images(x_res, dimensions, labeled = False, fignum = 0)
    
    plt.show()

def simplex_plot(model, x_test, y_test):
    
    if model.layers[-1].output_shape[-1] == 3:
        p_test = model.predict(x_test)
        p_test = func.softmax(p_test, axis = -1)
        p_test = p_test.T[: -1]
        
        T = np.array([[1, 1/2],
                      [0, np.sqrt(3)/2]])
        p_test = T @ p_test
        
        y_test = np.argmax(y_test, axis = 1)
        
        for y_cur in range(10):
            plt.figure(figsize = (20, 20*np.sqrt(3)/2))
            plt.hexbin(*p_test[:, y_cur == y_test], gridsize = (100, 50), bins = "log",
                       extent = [0, 1, 0, np.sqrt(3)/2], cmap = cmr.cosmic.reversed())
            plt.axis("scaled")
            plt.axis("off")
            plt.plot(np.array([0, 1, 1/2, 0]), np.array([0, 0, np.sqrt(3)/2, 0]), color = "black", zorder = 1)
            plt.xlim(0, 1)
            plt.ylim(0, np.sqrt(3)/2)
            plt.colorbar()
            plt.show()
    
    else:
        pass

def PCA_plot(model, x_train, x_test, y_test, number_classes):
    y_pred = model.predict(x_train)
    y_pred = y_pred - np.mean(y_pred, axis = 1, keepdims = True)
    y_pred = y_pred - np.mean(y_pred, axis = 0, keepdims = True)
    
    pca = np.linalg.svd(y_pred, full_matrices = False)[2][: 2].T
    
    y_pred = model.predict(x_test)
    y_pred = y_pred - np.mean(y_pred, axis = 1, keepdims = True)
    y_pred = y_pred - np.mean(y_pred, axis = 0, keepdims = True)
    
    y_proj = y_pred @ pca
    
    score = np.sum(np.var(y_proj, axis = 0)) / np.sum(np.var(y_pred, axis = 0))
    
    plt.scatter(*y_proj.T, marker = ".", c = y_test, s = 0.5, cmap = qualitative)
    plt.title(r"$%.2f$ of the variance explained" % score)
    # plt.contourf()
    
    number_classes = y_pred.shape[1] - 1
    cbar = plt.colorbar(ticks = np.linspace((1 - 1/number_classes)/2, number_classes - 1 - (1 - 1/number_classes)/2,
                                            num = number_classes, endpoint = True))
    cbar.ax.set_yticklabels(np.arange(number_classes))
    # cbar.ax.hlines
    cbar.ax.hlines(np.linspace(0, number_classes - 1, num = number_classes + 1, endpoint = True),
                   0, 1, color = "black", linewidths = 1)
    
    plt.show()
    return

### Animate lists of weights returned by WeightEvolution.
def animate(w_list, g_list, model):
    
    w_fig = plt.figure()
    w_images = []
    for w in w_list:
        w_image = plot_images(w, animated = True)
        w_images.append([w_image])
    
    movie = ArtistAnimation(w_fig, w_images, blit = True)
    movie.save("Data/Movies/w_movie.gif", writer = "pillow")
    plt.close()
    
    y_hard = np.argmax(model.predict(w), axis = 1)
    
    g_fig = plt.figure()
    g_images = []
    for g in g_list:
        g_image = plot_labels(y_hard, g, animated = True)
        # g_image = plot_labels(w, g, model, animated = True)
        g_images.append([g_image])
    
    movie = ArtistAnimation(g_fig, g_images, blit = True)
    movie.save("Data/Movies/g_movie.gif", writer = "pillow")
    plt.close()
