from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np

import DAN_code.normalization as norm

class InverseModel():
    def __init__(self, direct_model, softening = None):
        self.direct_model = direct_model
        
        self.softening = softening

    def preprocess_data(self, x_init, y_target):
        x_init = tf.cast(x_init, "float32")
        
        #y_target_rank = tf.rank(y_target)
        
        if self.softening is None:
            y_target = tf.cast(y_target, "bool")
        else:
            y_target = tf.cast(y_target, "float32")
            y_target = (1 - self.softening) * y_target + self.softening / (y_target.shape[1] + 1)
        
        if k.any(k.sum(tf.cast(y_target == k.max(y_target, axis = 1, keepdims = True), "float32"), axis = 1) > 1):
            raise ValueError("y_target should have a single maximum.")
        
        x_init = norm.tensor_normalize(x_init, norm.tensor_two_norm(x_init, axis = 1))
        
        return x_init, y_target
    
    def compute_loss(self, x_var, y_target):
        h_pred = self.direct_model(x_var)
        
        # L_tar = self.direct_model.compiled_loss(y_target, h_pred)
        
        if self.softening is None:
            L_tar = -tf.boolean_mask(h_pred[:, : -1], y_target)
        else:
            L_tar = -k.sum(y_target * h_pred[:, : -1], axis = 1) - self.softening / (y_target.shape[1] + 1) * h_pred[:, -1]
        
        c = k.stop_gradient(k.max(h_pred, axis = 1, keepdims = True))
        L_tar = L_tar + tf.squeeze(c) + k.log(k.sum(k.exp(h_pred - c), axis = 1))
        
        return L_tar, h_pred

    def compute_grad(self, x_var, y_target, x_init, inside_adversarial_sphere):
        with tf.GradientTape() as tape:
            tape.watch(x_var)
            L_tar, h_pred = self.compute_loss(x_var, y_target)
        
        grad = tape.gradient(L_tar, x_var)
        grad = grad - k.sum(grad * x_var, axis = 1, keepdims = True) * x_var
        
        x_perp = (x_var - x_init) - k.sum((x_var - x_init) * x_var, axis = 1, keepdims = True) * x_var
        x_perp = norm.tensor_normalize(x_perp, norm.tensor_two_norm(x_perp, axis = 1))
        
        grad = tf.where(inside_adversarial_sphere, grad, grad - k.sum(grad * x_perp, axis = 1, keepdims = True) * x_perp)
        
        # print(k.sum(grad * x_var, axis = 1))
        
        # return tape.gradient(L_tar, x_var), L_tar, h_pred
        return grad, L_tar, h_pred
    
    def smd(self, x_var, y_target, x_init, epsilon, learning_rate, momentum, velocity, grad, int_grad):
        
        velocity = momentum * velocity - learning_rate * grad
        
        x_var_new = x_var + learning_rate * velocity
        x_var_new = norm.tensor_normalize(x_var_new, norm.tensor_two_norm(x_var_new, axis = 1))
        
        adv_size = k.sum((x_var_new - x_init)**2, axis = 1, keepdims = True)**(1/2)
        
        inside_adversarial_sphere = adv_size <= epsilon
        
        x_var_new = tf.where(inside_adversarial_sphere, x_var_new, norm.tensor_subsphere_normalize(x_var_new, x_init, epsilon))
        
        grad_new, L_tar, h_pred = self.compute_grad(x_var_new, y_target, x_init, inside_adversarial_sphere)
        
        # print(k.max(x_var_new - x_var))
        int_grad = int_grad + (x_var_new - x_var) * (grad + grad_new)/2
        # int_grad = tf.where(inside_adversarial_sphere, int_grad + (x_var_new - x_var) * (grad + grad_new)/2, int_grad)
        
        # (x_2 - x_1) * grad_1 + (x_3 - x_2) * grad_2 + (x_4 - x_3) * grad_3
        # sum_n (x_(n+1) - x_n) grad_n = sum_n x_(n+1) grad_n - 
        
        # sum_n (x_(n+1) - x_n) (grad_n + grad_(n+1)) = sum_n x_(n+1) grad_n + sum_n x_(n+1) grad_(n+1) - sum_n x_n grad_n - sum_n x_n grad_(n+1) = sum_{n = 0}^{N - 1} x_(n+1) grad_n - sum_{n = 0}^{N - 1} x_n grad_(n+1) + x_N grad_N - x_0 grad_0
        # = sum_{n = 0}^{N - 1} x_(n+1) grad_n - sum_{n = 1}^{N} x_(n-1) grad_n + x_N grad_N - x_0 grad_0
        # = x_1 grad_0 - x_(N-1) grad_N + ...
        x_var = x_var_new
        grad = grad_new
        
        return x_var, velocity, L_tar, h_pred, grad, int_grad, adv_size
    
    def generate_data(self, x_init, y_target, learning_rate, momentum, number_epochs):
        x_init, y_target = self.preprocess_data(x_init, y_target)
        x_var = tf.identity(x_init)
        
        inside_adversarial_sphere = tf.constant(True, shape = (x_var.shape[0], 1))
        
        velocity = 0
        grad, L_init, h_pred = self.compute_grad(x_var, y_target, x_init, inside_adversarial_sphere)
        int_grad = 0
        
        epsilon = tf.constant(1., shape = (x_var.shape[0], 1))
        #epsilon = tf.constant(2., shape = (x_var.shape[0], 1))
        
        for current_epoch in range(number_epochs):
            x_var, velocity, L_tar, h_pred, grad, int_grad, adv_size = self.smd(x_var, y_target, x_init, epsilon, learning_rate, momentum, velocity, grad, int_grad)
            
            pred_matches_target = (k.argmax(h_pred, axis = 1) == k.argmax(y_target, axis = 1))[:, tf.newaxis]
            epsilon = tf.where(pred_matches_target & (epsilon == 1.), adv_size, epsilon)
        
        #for current_epoch in range(number_epochs):
        #    x_var, velocity, L_tar, h_pred, grad, int_grad = self.smd(x_var, y_target, x_init, epsilon, learning_rate, momentum, velocity, grad, int_grad)
        #
        #for current_bisection in range(number_bisections):
        #    pred_matches_target = (k.argmax(h_pred, axis = 1) == k.argmax(y_target, axis = 1))[:, tf.newaxis]
        #    shift = 0.5**(current_bisection + 1)
        #    
        #    epsilon = tf.where(pred_matches_target, epsilon - shift, epsilon + shift)
        #    
        #    if current_bisection == number_bisections - 1:
        #        epsilon = epsilon + shift
        #    
        #    for current_epoch in range(number_epochs):
        #        #y_target_rank = tf.rank(y_target)
        #        
        #        x_var, velocity, L_tar, h_pred, grad, int_grad = self.smd(x_var, y_target, x_init, epsilon, learning_rate, momentum, velocity, grad, int_grad)
        
        #current_epsilon = tf.constant(0., shape = (x_var.shape[0], 1))
        #for current_epoch in range(number_epochs):
        #    #current_epsilon = (current_epoch - 1) / number_epochs * epsilon
        #    #current_epsilon = epsilon
        #    
        #    #y_target_rank = tf.rank(y_target)
        #    
        #    current_epsilon = tf.where((k.argmax(h_pred, axis = 1) == k.argmax(y_target, axis = 1))[:, tf.newaxis], current_epsilon, current_epsilon + 1/number_epochs * epsilon)
        #    
        #    x_var, velocity, L_tar, h_pred, grad, int_grad = self.smd(x_var, y_target, x_init, current_epsilon, learning_rate, momentum, velocity, grad, int_grad)
        
        L_diff = L_tar - L_init
        
        x_final = x_var.numpy()
        int_grad = int_grad.numpy()
        
        print(epsilon)
        
        return x_final, L_diff, int_grad
    
    def perturb_data(self, x_init, y_target, epsilon, learning_rate, momentum, number_epochs):
        return self.generate_data(x_init, y_target, -learning_rate, momentum, number_epochs)