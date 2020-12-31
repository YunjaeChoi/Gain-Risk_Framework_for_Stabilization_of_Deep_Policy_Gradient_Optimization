import numpy as np
import tensorflow as tf

class LinearLR:

    def __init__(self, initial_lr, min_lr, delta_lr):

        self.lr_ph = tf.placeholder(tf.float32, shape=[])
        self.initial_lr = initial_lr
        self.lr = initial_lr
        self.min_lr = min_lr
        self.delta_lr = delta_lr

    def reset(self):
        self.lr = self.initial_lr

    def update(self):
        self.lr -= self.delta_lr
        self.lr = np.clip(self.lr, self.min_lr, self.initial_lr)
