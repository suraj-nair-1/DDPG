# Actor network for DDPG

import tensorflow as tf
import numpy as np
import tflearn
# from tensorflow import concat

class ActorNetworkReplay(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -2 and 2
    """
    def __init__(self, state_dim, action_dim, loadfrom):
        self.inputs = tflearn.input_data(shape=[None, state_dim])
        self.l = tflearn.fully_connected(self.inputs, 1024)
        self.la = tflearn.activations.leaky_relu(self.l, alpha=-.01)
        self.l2 = tflearn.fully_connected(self.la, 512)
        self.l2a = tflearn.activations.leaky_relu(self.l2, alpha=-.01)
        self.l3 = tflearn.fully_connected(self.l2a, 256)
        self.l3a = tflearn.activations.leaky_relu(self.l3, alpha=-.01)
        self.l4 = tflearn.fully_connected(self.l3a, 128)
        self.l4a = tflearn.activations.leaky_relu(self.l4, alpha=-.01)
        self.out = tflearn.fully_connected(self.l4a, action_dim)

        self.choice = tf.slice(self.out, [0,0], [-1, 4])
        self.params = tf.slice(self.out, [0,4], [-1, 6])
        self.choice_probs = tflearn.activations.softmax(self.choice)
        self.scaled_out = tflearn.merge([self.choice_probs, self.params], 'concat')
        self.model = tflearn.DNN(self.scaled_out)
        self.model.load(loadfrom)

    def model_load(self, loadfrom):
        self.model.load(loadfrom)

    def model_predict(self, inps):
        return self.model.predict(inps)
