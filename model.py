import numpy as np
import tensorflow as tf
from utils.math_utils import tanh_bounded_diag_normal, clip_but_pass_gradient
import tensorflow_probability as tfp
tfd = tfp.distributions

class Actor(tf.keras.Model):

    def __init__(self, action_shape, name='actor'):
        super(Actor, self).__init__(name=name)
        self.action_shape = action_shape
        self.action_size = np.prod(self.action_shape)

        #layers
        self.dense1 = tf.keras.layers.Dense(256)
        self.dense1_activation = tf.keras.layers.LeakyReLU()
        self.dense2 = tf.keras.layers.Dense(128)
        self.dense2_activation = tf.keras.layers.LeakyReLU()
        self.mu_layer = tf.keras.layers.Dense(self.action_size,
                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        self.std_layer = tf.keras.layers.Dense(self.action_size, activation=tf.keras.activations.softplus,
                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, inputs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = self.dense1(inputs)
            x = self.dense1_activation(x)
            x = self.dense2(x)
            x = self.dense2_activation(x)
            mu = self.mu_layer(x)
            std = self.std_layer(x)
            return mu, std


class VCritic(tf.keras.Model):

    def __init__(self, name='critic'):
        super(VCritic, self).__init__(name=name)
        #layers
        self.dense1 = tf.keras.layers.Dense(256)
        self.dense1_activation = tf.keras.layers.LeakyReLU()
        self.dense2 = tf.keras.layers.Dense(128)
        self.dense2_activation = tf.keras.layers.LeakyReLU()
        self.v_layer = tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, inputs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = self.dense1(inputs)
            x = self.dense1_activation(x)
            x = self.dense2(x)
            x = self.dense2_activation(x)
            v = self.v_layer(x)
            return v

def mlp_actor_critic(x, a_dn, next_x):
    state_input_keras = tf.keras.Input(tensor=x)
    action_dn_input_keras = tf.keras.Input(tensor=a_dn)
    next_state_input_keras = tf.keras.Input(tensor=next_x)

    state_shape = x.shape.as_list()[1:]
    action_shape = a_dn.shape.as_list()[1:]

    with tf.variable_scope('pi'):
        actor = Actor(action_shape)
        act_mu_dn, act_std_dn = actor(state_input_keras)
        act_dist = tfd.Normal(loc=act_mu_dn, scale=act_std_dn)

        pi_dn = act_dist.sample()
        logp_pi_dn_all = act_dist.log_prob(pi_dn)
        logp_pi_dn = tf.reduce_sum(logp_pi_dn_all, axis=1)
        pi, logp_pi = tanh_bounded_diag_normal(pi_dn, logp_pi_dn)

        logp_action_dn_all = act_dist.log_prob(action_dn_input_keras)
        logp_action_dn = tf.reduce_sum(logp_action_dn_all, axis=1)
        _, logp = tanh_bounded_diag_normal(action_dn_input_keras, logp_action_dn)

    with tf.variable_scope('v'):
        vcritic = VCritic()
        v = vcritic(state_input_keras)
        v = tf.squeeze(v, axis=1)

        next_v = vcritic(next_state_input_keras)
        next_v = tf.squeeze(next_v, axis=1)

    return pi, pi_dn, logp_pi, logp, act_mu_dn, act_std_dn, v, next_v
