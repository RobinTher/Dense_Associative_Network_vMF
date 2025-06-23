import numpy as np
import tensorflow as tf

import DAN_code.functions as func
import DAN_code.normalization as norm
from DAN_code import layers
from DAN_code.inverse_model import InverseModel

import matplotlib.pyplot as plt

import DAN_code.result_plotting as res_plot

### Sort prototypes by recognized classes and plot
def sort_prototypes(model, number_memories):
    
    w = model.get_DAN_layer(1).get_weights()[0].T
    g = model.get_DAN_layer(2).get_weights()[0]
    
    y_hard = np.argmax(model.predict(w), axis = 1)
    j_sorted = np.argsort(y_hard)
    w_sorted = w[j_sorted]
    y_sorted = y_hard[j_sorted]
    g_sorted = g[j_sorted]
    
    # Plot samples
    for j in range(number_memories // 25):
        res_plot.plot_images(w_sorted[25 * j : 25 * (j + 1)])
        res_plot.plot_labels(y_sorted[25 * j : 25 * (j + 1)], g_sorted[25 * j : 25 * (j + 1)].T)

### Plot the memories which are the most activated by each image in x_test and print their classes
def top_activated_memories(x_test, model, x_init = None):
    
    w = model.get_DAN_layer(1).get_weights()[0]
    g = model.get_DAN_layer(2).get_weights()[0]
    
    y_pred = np.argmax(model.predict(x_test), axis = -1)
    y_1NN, y_2NN = first_neighbor(x_test, w, g)
    
    beta = model.get_DAN_layer(1).beta
    tau = model.get_DAN_layer(2).tau
    
    counts_memory = np.squeeze(model.get_DAN_layer(2).counts_memory)
    
    x = x_test - np.mean(x_test, axis = 1, keepdims = True)
    h = beta * func.dense_cor(x, w)
    
    p_h = func.softmax(h + np.log(counts_memory[: -1]), tau + np.log(counts_memory[-1 :]), axis = -1)
    res_plot.plot_images(p_h @ w.T)
    
    for p_h_cur, y_pred_cur, y_1NN_cur in zip(p_h, y_pred, y_1NN):
        j_top = np.flip(np.argsort(p_h_cur))[: 25]
        
        w_top = w[:, j_top].T
        g_top = g[j_top].T
        
        p_h_top = p_h_cur[j_top]
        
        title = r"Network prediction: $y = %d$, 1NN approximation: $y = %d$" % (y_pred_cur, y_1NN_cur)
        
        res_plot.plot_activations(p_h_top, w_top, g_top, title = title)
        
def linear_decomposition(x_final, x_init, y_target, model, softening):
    #target_y = np.array(y_target, ndmin = 1)
    
    w = model.get_DAN_layer(1).get_weights()[0]
    g = model.get_DAN_layer(2).get_weights()[0]
    
    counts_memory = model.get_DAN_layer(2).counts_memory.numpy()
    
    beta = model.get_DAN_layer(1).beta
    tau = model.get_DAN_layer(2).tau
    
    h = beta * func.dense_cor(x_final, w)
    
    #prob_memory_given_target = func.weighed_softmax(h, g, tau, y_target)
    #prob_memory = 1/(1 - softening) * func.weighed_softmax(h, counts_memory, tau) - softening/(1 - softening) / (y_target.shape[1] + 1) * func.weighed_softmax(h, g, tau)
    
    y_target = (1 - softening) * y_target + softening / (y_target.shape[1] + 1)
    prob_memory_given_target = func.weighed_softmax(h, g, tau, y_target)
    prob_memory = func.weighed_softmax(h, counts_memory, tau)
    
    w_mean_given_target = prob_memory_given_target @ w.T
    w_mean = prob_memory @ w.T
    
    w_diff = w_mean_given_target - w_mean
    
    w_diff = norm.array_normalize(w_diff, norm.array_two_norm(w_diff, axis = 1))
    
    w_diff_x_init_overlap = np.sum(w_diff*x_init, axis = 1, keepdims = True)
    
    w_diff_x_final_overlap = np.sum(w_diff*x_final, axis = 1, keepdims = True)
    x_init_x_final_overlap = np.sum(x_final*x_init, axis = 1, keepdims = True)
    
    w_diff_coefs = w_diff_x_final_overlap - w_diff_x_init_overlap * x_init_x_final_overlap
    x_init_coefs = x_init_x_final_overlap - w_diff_x_init_overlap * w_diff_x_final_overlap
    
    magn = np.sum((w_diff_coefs * w_diff + x_init_coefs * x_init)**2, axis = 1, keepdims = True)**(1/2)
    w_diff_coefs = w_diff_coefs / magn
    x_init_coefs = x_init_coefs / magn
    
    x_recon = w_diff_coefs * w_diff + x_init_coefs * x_init
    
    x_res = x_final - x_recon
    x_res_coefs = norm.array_two_norm(x_res, axis = 1)
    x_res = norm.array_normalize(x_res, x_res_coefs)
    
    w_mean_given_target_coefs = norm.array_two_norm(w_mean_given_target, axis = 1)
    w_mean_given_target = norm.array_normalize(w_mean_given_target, w_mean_given_target_coefs)
    w_mean_given_target_coefs = w_mean_given_target_coefs * w_diff_coefs
    
    w_mean_coefs = norm.array_two_norm(w_mean, axis = 1)
    w_mean = norm.array_normalize(w_mean, w_mean_coefs)
    w_mean_coefs = w_mean_coefs * w_diff_coefs
    
    for i in range(len(h)):
        res_plot.plot_linear_decomposition(x_final[i], x_init[i], w_mean_given_target[i], w_mean[i], x_res[i],
                                           x_init_coefs[i], w_mean_given_target_coefs[i], w_mean_coefs[i], x_res_coefs[i])
        
        j_top = np.flip(np.argsort(prob_memory_given_target[i]))[: 25]
        
        w_top = w[:, j_top].T
        g_top = g[j_top].T
        
        prob_memory_given_target_top = prob_memory_given_target[i][j_top]
        
        res_plot.plot_activations(prob_memory_given_target_top, w_top, g_top)
        
        ###
        
        j_top = np.flip(np.argsort(prob_memory[i]))[: 25]
        
        w_top = w[:, j_top].T
        g_top = g[j_top].T
        
        prob_memory_top = prob_memory[i][j_top]
        
        res_plot.plot_activations(prob_memory_top, w_top, g_top)

def weighted_median(values, weights, quantile = 1/2):
    
    indices = np.argsort(values, axis = -1)
    
    values = np.take_along_axis(values, indices, axis = -1)
    weights = np.take_along_axis(weights, indices, axis = -1)
    
    cum_weights = np.cumsum(weights, axis = -1)
    
    #indices = np.array([np.searchsorted(cum_weight, quantile * cum_weight[-1]) for cum_weight in cum_weights])
    #
    #return indices, np.array([value[index] for value, index in zip(values, indices)])
    
    return np.array([value[np.searchsorted(cum_weight, quantile * cum_weight[-1])] for value, cum_weight
                     in zip(values, cum_weights)])

def generate_and_plot_data(x_init, y_target, model, softening, learning_rate, momentum, number_epochs):
    
    inverse_model = InverseModel(model, softening)
    x_final, L_diff, int_grad = inverse_model.generate_data(x_init, y_target, learning_rate, momentum, number_epochs)
    #print(y_target)
    
    # Plot initial conditions and generated data
    # Also plot the weights which are the most activated by clean and perturbed images
    res_plot.plot_images(x_init)
    # top_activated_memories(x_init, model)
    
    res_plot.plot_images(x_final)
    
    L_diff_deviations = np.abs((L_diff - np.sum(int_grad, axis = 1)) / L_diff)
    L_diff_error = np.mean(L_diff_deviations)
    
    res_plot.plot_images(int_grad, title = r"Integrated gradient relative error $\approx %.2f$" % L_diff_error)
    
    return x_final

### Attack all digits fed to the models and plot the accuracy as a function of the attack magnitude
# def fool_all(x_test, y_test, models):
    
    # for model in models:
        # Set up foolbox
        # epsilons = np.linspace(0, 1, num = 11)
        # attack = foolbox.attacks.LinfProjectedGradientDescentAttack()
        # fmodel = foolbox.models.TensorFlowModel(model, bounds = (0, 1), preprocessing = None)
        
        # x_clean = tf.convert_to_tensor(x_test)
        # y_clean = tf.convert_to_tensor(np.argmax(y_test, axis = 1))
        
        # criterion = foolbox.criteria.Misclassification(y_clean)
        
        # x_raw, x_adv, is_adv = attack(fmodel, x_clean, criterion = criterion, epsilons = epsilons)
    
        # plt.plot(epsilons, 1 - is_adv.numpy().mean(axis = -1), label = r"$\beta = $" + str(model.layers[-1].beta))
        # plt.legend()
        # plt.xlabel("Magnitude of adversarial perturbations")
        # plt.ylabel("Accuracy of the network")
    
    # plt.show()

def first_neighbor(x, w, g):
    
    h = func.dense_cor(x - np.mean(x, axis = 1, keepdims = True), w)
    
    y_1NN = np.argmax(g[np.argmax(h, axis = 1)], axis = 1)
    
    h[:, np.argmax(h, axis = 1)] = 0
    
    y_2NN = np.argmax(g[np.argmax(h, axis = 1)], axis = 1)
    
    return y_1NN, y_2NN
    # return np.argmax(g[np.argmax(h, axis = 1)], axis = 1)

def first_neighbor_fidelity(x_test, y_hard, y_pred, w, g):
    y_1NN, y_2NN = first_neighbor(x_test, w, g)
    
    print("1NN score: " + str(np.mean(y_1NN == y_hard)))
    print("1NN fidelity: " + str(np.mean(y_1NN == y_pred)))
    
    print("Residual 2NN fidelity: " + str(np.mean(y_2NN[y_1NN != y_pred] == y_pred[y_1NN != y_pred])))
    
    idx_pass = y_pred == y_hard
    idx_fail = ~idx_pass
    
    print("1NN fidelity on pass: " + str(np.mean(y_1NN[idx_pass] == y_pred[idx_pass])))
    print("1NN fidelity on fail: " + str(np.mean(y_1NN[idx_fail] == y_pred[idx_fail])))

### Plot some misclassified digits
def misclassified_digits(x_test, y_hard, model):
    p_hard = np.argmax(model.predict(x_test), axis = 1)
    x_miss = x_test[p_hard != y_hard]
    for i in range(x_miss.shape[0] // 25):
        res_plot.plot_images(x_miss[25 * i : 25 * (i + 1)])