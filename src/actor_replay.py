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
        # print self.model.get_weights(self.l.W)
        self.model.load(loadfrom)
        # print self.model.get_weights(self.l.W)

    def model_load(self, loadfrom):
        self.model.load(loadfrom)

    def model_predict(self, inps):
        return self.model.predict(inps)

    def add_noise(self, a, eps):
        if (np.random.random_sample() <= eps) or (np.isnan(a).any()):
            # print "RANDOM AF &&&&&&&&&&&&&&&&&"
            acts = np.random.uniform(1, 10, 4)
            a[:4] = acts / np.sum(acts)
            a[4] = np.random.uniform(0, 100)
            a[5] = np.random.uniform(-180, 180)
            a[6] = np.random.uniform(-180, 180)
            a[7] = np.random.uniform(-180, 180)
            a[8] = np.random.uniform(0, 100)
            a[9] = np.random.uniform(-180, 180)
        else:
            print a

        index = np.argmax(a[:4])

        return index, a