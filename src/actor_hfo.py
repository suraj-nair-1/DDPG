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
    def __init__(self, sess, state_dim, action_dim, low_action_bound, high_action_bound, learning_rate, tau, LOGPATH):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.low_action_bound = low_action_bound
        self.high_action_bound = high_action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.LOGPATH = LOGPATH
        # self.ou_noise_params = ou_noise_params
        # self.ou_noise = [r[1] for r in self.ou_noise_params]

        # Actor Network
        self.inputs, self.out, self.scaled_out, self.model = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out, self.target_model = self.create_actor_network()

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
        l = tflearn.fully_connected(inputs, 1024)
        la = tflearn.activations.leaky_relu(l, alpha=-.01)
        l2 = tflearn.fully_connected(la, 512)
        l2a = tflearn.activations.leaky_relu(l2, alpha=-.01)
        l3 = tflearn.fully_connected(l2a, 256)
        l3a = tflearn.activations.leaky_relu(l3, alpha=-.01)
        l4 = tflearn.fully_connected(l3a, 128)
        l4a = tflearn.activations.leaky_relu(l4, alpha=-.01)
        w_init = tflearn.initializations.normal(stddev = -0.01)
        out = tflearn.fully_connected(l4a, self.a_dim, weights_init=w_init)

        choice = tf.slice(out, [0,0], [-1, 4])
        params = tf.slice(out, [0,4], [-1, 6])
        choice_probs = tflearn.activations.softmax(choice)

        # params2 = tf.multiply(tf.divide(params + 1, 2), self.high_action_bound - self.low_action_bound) + self.low_action_bound
        # print tf.concat([tflearn.activations.sigmoid(choice), \
        #     tf.mul(params, self.high_action_bound - self.low_action_bound) + self.low_action_bound], 0)
        # print "***************"
        # print choice_probs
        # print params2
        # print params2
        # print choice_probs
        scaled_out = tflearn.merge([choice_probs, params], 'concat')
        model = tflearn.DNN(scaled_out, session = self.sess)
        # scaled_out = tf.concat(0, [choice_probs, params])
        # print scaled_out

        # Scale output to low_action_bound to high_action_bound
        # scaled_out = tf.div(out, tf.reduce_sum(out))
        # scaled_out = tf.mul(out, self.high_action_bound - self.low_action_bound) + self.low_action_bound
        return inputs, out, scaled_out, model

    # def load(self, loadfrom):
    #     if loadfrom is not None:
    #         self.model.load(loadfrom)
    #         self.target_model.load(loadfrom)

    def save_model(self, iterationnum):
        # model = tflearn.DNN(self.target_scaled_out, session = self.sess)
        # print model
        self.target_model.save(self.LOGPATH + "models/actor_run10_" + str(iterationnum)+".tflearn")

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
        # self.ou_noise_params = ou_noise_params
        # self.ou_noise = [r[1] for r in self.ou_noise_params]
        # Update OU noise for continuous params
        # for i in range(len(self.ou_noise)):
        #     dWt = np.random.normal(0.0, 1.0)
        #     [theta, mu, sigma] = self.ou_noise_params[i]
        #     self.ou_noise[i] += theta * (mu - self.ou_noise[i]) * 1.0 + sigma * dWt
        # print "NOISE", self.ou_noise, "EPSILON", eps'
        # if np.random.uniform() < 0.01:
        #     f = open(self.LOGPATH + 'noiselogs4.txt', 'a')
        #     f.write(str(self.ou_noise)[1:-1] + "\n")
        #     f.close()

        # Update continuous params
        # for i in range(len(self.ou_noise)):
        #     a[4 + i] += self.ou_noise[i]

        # With probability epsilon, set all discrete actions to be equally likely

        if np.random.random_sample() <= eps:
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
