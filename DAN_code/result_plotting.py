import numpy as np

import DAN_code.functions as func
import DAN_code.normalization as norm

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.animation import ArtistAnimation
import cmasher as cmr
import seaborn as sns

import string
uppercase_array = np.array(list(string.ascii_uppercase), dtype = "str")

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def truncate_colormap(cmap, minval = 0, maxval = 1, n = 256):
    '''
    Truncate a colormap to the range [minval, maxval] and return the result.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        The colormap to truncate.
    minval : float, optional
        The minimum value of the colormap range. Defaults to 0.
    maxval : float, optional
        The maximum value of the colormap range. Defaults to 1.
    n : int, optional
        The number of colors in the truncated colormap. Defaults to 256.
    Returns
    -------
    new_cmap : matplotlib.colors.Colormap
        The truncated colormap.
    '''
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

def plot_images(images, dimensions = None, animated = False, title = None, labeled = True, filename = None, fignum = None, fontsize = 10):
    '''
    Plot images in a grid. The images are reshaped to fit the grid dimensions.
    If the images do not fill all the squares of the grid, it is padded with zeros.
    If the images are given as 1D arrays, they are reshaped to 2D before plotting.

    Parameters
    ----------
    images : np.ndarray
        The images to plot. Can be 1D for a single image given as a 1D array, 2D for multiple
        images given as 1D arrays, or 3D/4D for multiple images with height and width.
        In the latter case, the last two dimensions are interpreted as the height and width
        of the images, while the other dimensions are interpreted as containing the
        different images.
    dimensions : tuple, optional
        The dimensions of the grid to plot the images. If not provided, the grid is
        inferred from the number of images. Defaults to None.
    animated : bool, optional
        If True, gives the option to use the plotted weights in an animation.
        Otherwise, plots the images in a static figure. Defaults to False.
    title : str, optional
        Optional plot title. Defaults to None for no title.
    labeled : bool, optional
        Whether to label the columns and rows of the grid with letters from A to Z.
        Defaults to True.
    filename : str, optional
        If provided, saves the plot to a file with the given filename.
        Defaults to None for no file saving.
    fignum : int, optional
        The figure number to plot the images on. If None, a new figure is created.
        Defaults to None.
    fontsize : int, optional
        The font size of the labels, the ticks and the optional title. Defaults to 10.
    Returns
    -------
    height : int
        The height of the grid in number of images. Returned iff animated is False.
    width : int
        The width of the grid in number of images. Returned iff animated is False.
    image : matplotlib.image.AxesImage
        The image object containing the plotted images. Returned iff animated is True.
    '''
    if images.ndim == 1:
        images = images[np.newaxis]
    
    if animated and (fignum is None):
        # The image is put onto the current figure if animated
        fignum = 0
    
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
                   labels = uppercase_array[np.arange(width).astype("int") % 26], fontsize = fontsize)
        
        plt.yticks(ticks = np.arange(0.5 * image_height - 0.5, image_height * (height + 0.5) - 0.5, image_height),
                   labels = uppercase_array[np.arange(height).astype("int") % 26], fontsize = fontsize)
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
            plt.title(title, fontsize = fontsize)
        
        if labeled:
            cbar = plt.colorbar(ticks = [-cap, 0, cap], shrink = 0.8, aspect = 20*0.8)
            cbar.ax.tick_params(labelsize = fontsize)
        
        if filename is not None:
            plt.savefig("./Data/Figures/%s.png" % filename)
        
        if fignum is None:
            plt.show()
        
        return height, width

def plot_labels(predictions, labels, dimensions = None, animated = False, labeled = True, filename = None, fignum = None, fontsize = 10):
    '''
    Plot soft label-like data corresponding to known images. For example,
    plot the predicted probabilities of a neural network on test images, the true labels
    of an image dataset, or the class weights of a DAN. We use this function to plot
    the class weights in our paper.

    Parameters
    ----------
    predictions : np.ndarray
        The classes predicted by a neural network fed with the known images. Must be 1D
        and consist of integers. When we use this function to plot the class weights of a DAN,
        we use the predictions of the DAN on its memories.
    labels : np.ndarray
        The label-like data to plot. Must be 2D, where the first dimension corresponds to the
        classes and the second dimension corresponds to the images.
    dimensions : tuple, optional
        The dimensions of the grid in which the known images were plotted. If not provided, the grid is
        inferred from the shape of the predictions. Defaults to None.
    animated : bool, optional
        If True, gives the option to use the plotted labels in an animation.
        Otherwise, plots the labels in a static figure. Defaults to False.
    labeled : bool, optional
        Whether to label the columns of the label-like data with tuples of letters from A to Z
        corresponding to the columns and rows of an image grid plotted for the known images
        using plot_images. Defaults to True.
    filename : str, optional
        If provided, saves the plot to a file with the given filename.
        Defaults to None for no file saving.
    fignum : int, optional
        The figure number to plot the labels on. If None, a new figure is created.
        Defaults to None.
    fontsize : int, optional
        The font size of the labels and ticks. Defaults to 10.
    Returns
    -------
    image : matplotlib.image.AxesImage
        The image object containing the plotted labels. Returned iff animated is True.
    '''
    if animated & (fignum is None):
        # The image is put onto the current figure if animated
        fignum = 0
    
    # Infer image dimensions if not provided
    number_images = predictions.shape[0]
    if dimensions == None:
        height = int(np.ceil(number_images**(1/2)))
        width = height
    else:
        height = dimensions[0]
        width = dimensions[1]
    
    cap = norm.array_max_norm(labels, keepdims = False)
    
    image = plt.matshow(labels, fignum = fignum, cmap = cold, vmin = 0, vmax = cap, animated = animated)
    plt.vlines(np.arange(0.5, labels.shape[1] - 0.5), -0.5, labels.shape[0] - 0.5, color = "black",
               linestyles = "dashed", linewidths = 1)
    
    if labeled:
        plt.xlabel("Memory", fontsize = fontsize)
        plt.ylabel("Class", fontsize = fontsize)
        
        tick_locations, tick_labels = plt.yticks(np.arange(labels.shape[0] - 1), fontsize = fontsize)
        
        height_labels = np.arange(height).astype("int")
        width_labels = np.arange(width).astype("int")
        
        height_labels = uppercase_array[height_labels % 26]
        width_labels = uppercase_array[width_labels % 26]
        
        memory_indices = np.array(np.meshgrid(height_labels, width_labels)).reshape(2, -1)
        memory_indices = np.char.add(memory_indices[1], memory_indices[0])
        
        plt.xticks(np.arange(labels.shape[1])[: number_images], labels = memory_indices[: number_images], rotation = 45, fontsize = fontsize)
        
        main_axis = plt.gca()
        
        main_axis.xaxis.tick_bottom()
        
        secondary_axis = main_axis.secondary_xaxis("top")
        
        secondary_axis.set_xticks(np.arange(labels.shape[1])[: number_images])
        secondary_axis.set_xticklabels(predictions.astype("int8"), fontsize = fontsize)
    else:
        plt.tick_params(top = False, bottom = False, left = False, right = False,
                        labeltop = False, labelbottom = False,  labelleft = False, labelright = False)
    
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

def plot_accuracy_and_runtime(beta, number_memories_range, number_splits_range, fontsize = 13):
    '''
    Plot the accuracy and run time of DANs trained with and without splitting as a function
    of the maximum number of memories. Assume that the accuracy and training time data
    is stored in Data/Performance with file names of the form
    DAN_accuracy_and_run_time_with_beta={beta}_and_{number_memories}_memories_for_{number_splits}_splits.npy,
    where beta and number_memories are the inverse temperature and maximum number of memories
    of the corresponding DANs, respectively, and number_splits is the number of splits of the DANs
    trainined with splitting steepest descent.

    Parameters
    ----------
    beta : float
        The inverse temperature of the DANs whose accuracy and run time to plot.
    number_memories_range : list of int
        The range of maximum number of memories for which to plot the accuracy and run time
        of the DANs.
    number_splits_range : list of int
        The range of number of splits for which to plot the accuracy and run time of the DANs
        trained with splitting steepest descent. DANs trained without splitting have
        number_splits = 0. This list must have the same length as number_memories_range.
    fontsize : int, optional
        The font size of labels and ticks. Defaults to 13.
    '''
    run_time_with_splits = []
    accuracy_with_splits = []
    run_time_without_splits = []
    accuracy_without_splits = []
    for number_memories, number_splits in zip(number_memories_range, number_splits_range):
        with open("./Data/Performance/DAN_accuracy_and_run_time_with_beta=%s_and_%s_memories_for_%s_splits.npy"
                % (str(beta), str(number_memories), str(number_splits)), "rb") as f:
            run_time = np.load(f)
            accuracy = np.load(f)
        
        run_time_with_splits.append(run_time)
        accuracy_with_splits.append(accuracy)
        
        with open("./Data/Performance/DAN_accuracy_and_run_time_with_beta=%s_and_%s_memories_for_%s_splits.npy"
                % (str(beta), str(number_memories), str(0)), "rb") as f:
            run_time = np.load(f)
            accuracy = np.load(f)
        
        run_time_without_splits.append(run_time)
        accuracy_without_splits.append(accuracy)

    scaling_factor = np.mean(run_time_with_splits[-1])

    run_time_with_splits = np.array(run_time_with_splits / scaling_factor)
    accuracy_with_splits = np.array(accuracy_with_splits)

    mean_run_time_with_splits = np.mean(run_time_with_splits, axis = 1)
    std_run_time_with_splits = np.std(run_time_with_splits, axis = 1)
    mean_accuracy_with_splits = np.mean(accuracy_with_splits, axis = 1)
    std_accuracy_with_splits = np.std(accuracy_with_splits, axis = 1)

    run_time_without_splits = np.array(run_time_without_splits / scaling_factor)
    accuracy_without_splits = np.array(accuracy_without_splits)

    mean_run_time_without_splits = np.mean(run_time_without_splits, axis = 1)
    std_run_time_without_splits = np.std(run_time_without_splits, axis = 1)
    mean_accuracy_without_splits = np.mean(accuracy_without_splits, axis = 1)
    std_accuracy_without_splits = np.std(accuracy_without_splits, axis = 1)

    fig, axes = plt.subplots(1, 2, sharex = True, figsize = (8, 4))
    fig_axis = fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = "none", which = "both", top = False, bottom = False, left = False, right = False)

    axes[0].errorbar(number_memories_range, mean_accuracy_with_splits, std_accuracy_with_splits,
                    marker = "o", linestyle = "--", capsize = 2, zorder = 2.5, color = "C0",
                    markersize = 4, label = "Splitting")
    axes[0].errorbar(number_memories_range, mean_accuracy_without_splits, std_accuracy_without_splits,
                    marker = "o", linestyle = "--", capsize = 2, zorder = 1.5, color = "C1",
                    markersize = 4, label = "No splitting")
    axes[0].legend(fontsize = fontsize)
    axes[0].set_xticks(number_memories_range)
    axes[0].set_ylabel(r"Accuracy", fontsize = fontsize)
    axes[0].tick_params(axis = "both", which = "both", labelsize = fontsize)

    axes[1].errorbar(number_memories_range, mean_run_time_with_splits, std_run_time_with_splits,
                    marker = "o", linestyle = "--", capsize = 2, zorder = 2.5, color = "C0", markersize = 4)
    axes[1].errorbar(number_memories_range, mean_run_time_without_splits, std_run_time_without_splits,
                    marker = "o", linestyle = "--", capsize = 2, zorder = 1.5, color = "C1", markersize = 4)
    axes[1].set_xticks(number_memories_range)
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].set_ylabel(r"Training time", fontsize = fontsize, rotation = -90, labelpad = 17)
    axes[1].tick_params(axis = "both", which = "both", labelsize = fontsize)
    axes[1].annotate(r"$t = 1$", (2000, 1), (1775, 1.25), fontsize = fontsize,
                    arrowprops = {"width" : 2, "headwidth" : 8, "headlength" : 8,
                                "shrink" : 1, "color" : "black"})

    fig_axis.set_xlabel(r"Max number memories $P_{\mathrm{max}}$", fontsize = fontsize)
    plt.show()

    fig, axes = plt.subplots(1, 2, sharex = True, figsize = (8, 4))
    fig_axis = fig.add_subplot(111, frameon = False)
    plt.tick_params(labelcolor = "none", which = "both", top = False, bottom = False, left = False, right = False)

    axes[0].plot(number_memories_range, mean_accuracy_with_splits,
                marker = "none", linestyle = "--", zorder = 2.5, color = "C0",
                label = "Splitting")
    axes[0].plot(number_memories_range, mean_accuracy_without_splits,
                marker = "none", linestyle = "--", zorder = 1.5, color = "C1",
                label = "No splitting")

    axes[0].plot(number_memories_range, accuracy_with_splits, marker = "o",
                markersize = 2, linestyle = "none", zorder = 2.5, color = "C0")
    axes[0].plot(number_memories_range, accuracy_without_splits,
                markersize = 2, marker = "o", linestyle = "none", zorder = 1.5, color = "C1")

    axes[0].legend(fontsize = fontsize)
    axes[0].set_xticks(number_memories_range)
    axes[0].set_ylabel(r"Accuracy", fontsize = fontsize)
    axes[0].tick_params(axis = "both", which = "both", labelsize = fontsize)

    axes[1].plot(number_memories_range, mean_run_time_with_splits,
                marker = "none", linestyle = "--", zorder = 2.5, color = "C0")
    axes[1].plot(number_memories_range, mean_run_time_without_splits,
                marker = "none", linestyle = "--", zorder = 1.5, color = "C1")

    axes[1].plot(number_memories_range, run_time_with_splits,
                marker = "o", markersize = 2, linestyle = "none", zorder = 2.5, color = "C0")
    axes[1].plot(number_memories_range, run_time_without_splits,
                marker = "o", markersize = 2, linestyle = "none", zorder = 1.5, color = "C1")
    axes[1].set_xticks(number_memories_range)
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].set_ylabel(r"Training time", fontsize = fontsize, rotation = -90, labelpad = 17)
    axes[1].tick_params(axis = "both", which = "both", labelsize = fontsize)

    fig_axis.set_xlabel(r"Max number memories $P_{\mathrm{max}}$", fontsize = fontsize)
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
    '''
    Create an animation of the evolution of DAN memories and class weights during training.

    Parameters
    ----------
    w_list : list of np.ndarray
        A list of DAN memories at different training steps.
    g_list : list of np.ndarray
        A list of DAN class weights at different training steps.
    model : keras.Model
        The trained DAN model used to predict the hard class labels of the memories,
        which are used as the predictions argument of plot_labels.
    '''
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
