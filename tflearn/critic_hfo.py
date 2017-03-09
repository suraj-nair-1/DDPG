# Critic network for DDPG

import tensorflow as tf
import numpy as np
import tflearn

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, action_dim, low_act_bound, high_act_bound, learning_rate, tau, num_actor_vars, MINIBATCH_SIZE):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action (i.e., sum of dy/dx over all ys). We then divide
        # through by the minibatch size to scale the gradients down correctly.
        n2 = tf.div(tf.gradients(self.out, self.action), tf.constant(MINIBATCH_SIZE, dtype=tf.float32))
        # print n2
        # print self.action
        choice = tf.slice(n2, [0,0,0], [-1, -1, 4])
        params = tf.slice(n2, [0,0,4], [-1, -1, 6])

        high = tf.constant([[high_act_bound.tolist()]*MINIBATCH_SIZE])
        # print high
        low = tf.constant([[low_act_bound.tolist()]*MINIBATCH_SIZE])
        # print high - params
        pmax = tf.div((high - params), (high - low))
        pmin = tf.div((params - low), (high - low))
        # print pmax

        comparison = tf.less(tf.constant(0.0), params)

        # print comparison
        self.action_grads =  tflearn.merge([choice, tf.multiply(params, tf.where(comparison, pmax, pmin))], axis = 2, mode ='concat')
        print self.action_grads


    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        # net2 = tflearn.activation(tf.matmul(net,t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        net2 = tflearn.activation(tflearn.merge([tf.matmul(net,t1.W), tf.matmul(action, t2.W)], 'concat'), activation='relu')
        # tflearn.merge([inputs, action], 'concat')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net2, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.loss, self.optimize], feed_dict={
        # return self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    # def getloss(self, inputs, action, predicted_q_value):
    #     return self.sess.run([self.out,self.loss], feed_dict={
    #     # return self.sess.run(self.optimize, feed_dict={
    #         self.inputs: inputs,
    #         self.action: action,
    #         self.predicted_q_value: predicted_q_value
    #     })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
