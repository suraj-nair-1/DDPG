# Actor network for DDPG

import tensorflow as tf
import numpy as np
import tflearn
# from tensorflow import concat

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -2 and 2
    """
    def __init__(self, sess, state_dim, action_dim, low_action_bound, high_action_bound, learning_rate, tau, ou_noise_params):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.low_action_bound = low_action_bound
        self.high_action_bound = high_action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.ou_noise_params = ou_noise_params
        self.ou_noise = [r[1] for r in self.ou_noise_params]

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + \
                tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')
        net2 = tflearn.fully_connected(net, 300, activation='relu')
        # TODO: Final layer weights need to be initted to their ranges
        w_init = tflearn.initializations.uniform(minval=-.003, maxval=.003)
        out = tflearn.fully_connected(net2, self.a_dim, activation='tanh', weights_init=w_init)
        # print out
        choice = tf.slice(out, [0,0], [-1, 4])
        params = tf.slice(out, [0,4], [-1, 6])
        # print choice
        # print params
        choice_probs = tflearn.activations.softmax(choice)
        # print choice_probs
        params2 = tf.multiply(tf.divide(params + 1, 2), self.high_action_bound - self.low_action_bound) + self.low_action_bound
        # print tf.concat([tflearn.activations.sigmoid(choice), \
        #     tf.mul(params, self.high_action_bound - self.low_action_bound) + self.low_action_bound], 0)
        # print "***************"
        # print choice_probs
        # print params2
        scaled_out = tflearn.merge([choice_probs, params2], 'concat')
        # print scaled_out

        # Scale output to low_action_bound to high_action_bound
        # scaled_out = tf.div(out, tf.reduce_sum(out))
        # scaled_out = tf.mul(out, self.high_action_bound - self.low_action_bound) + self.low_action_bound
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs,
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def add_noise(self, a, eps):
        # Update OU noise for continuous params
        for i in range(len(self.ou_noise)):
            dWt = np.random.normal(0.0, 1.0)
            [theta, mu, sigma] = self.ou_noise_params[i]
            self.ou_noise[i] += theta * (mu - self.ou_noise[i]) * 1.0 + sigma * dWt
        print "NOISE", self.ou_noise

        # Update continuous params
        for i in range(len(self.ou_noise)):
            a[4 + i] += self.ou_noise[i]

        # With probability epsilon, set all discrete actions to be equally likely
        if np.random.random_sample() <= eps:
            # print "RANDOM AF &&&&&&&&&&&&&&&&&"
            acts = np.random.uniform(1, 10, 4)
            a[:4] = acts / np.sum(acts)
        index = np.argmax(a[:4])

        return index, a
