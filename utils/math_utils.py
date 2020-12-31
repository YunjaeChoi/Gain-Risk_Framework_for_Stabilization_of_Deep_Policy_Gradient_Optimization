import numpy as np
import tensorflow as tf
import scipy.signal

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def diag_normal_entropy(mean, std):
    sum_log_std = tf.math.reduce_sum(tf.math.log(std), axis=1)
    ent = sum_log_std + 0.5 * mean.shape[1].value * tf.math.log(2.0 * np.pi * np.e)
    #return tf.reduce_mean(ent)
    return ent

def diag_normal_kl_divergence(mean0, std0, mean1, std1):
    """
    D_KL(diag_normal0 || diag_normal1)
    """
    diag_cov0 = std0**2
    diag_cov1 = std1**2
    diag_trace = tf.math.reduce_sum(diag_cov0 / diag_cov1, axis=1)
    mc = tf.math.reduce_sum((mean1 - mean0)**2 / diag_cov1, axis=1)
    log_det_ratio = tf.math.reduce_sum(tf.math.log(diag_cov1 / diag_cov0), axis=1)
    kl = 0.5 * (diag_trace + mc + log_det_ratio - mean0.shape[1].value)
    #return tf.reduce_mean(kl)
    return kl

def clip_but_pass_gradient(x, l=-1.0, u=1.0, epsilon=1e-5):
    l += epsilon
    u -= epsilon
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

def tanh_bounded_diag_normal(z, logp_z):
    """
    input:
        [samples, log_prob(samples)] from diagonal normal

    output:
        [tanh(z), adjusted log_prob]
    """
    logp_z -= tf.math.reduce_sum(2.0*(np.log(2.0) - z - tf.nn.softplus(-2.0*z)), axis=1)
    z = tf.tanh(z)
    return z, logp_z
