# Critic network for DDPG

import tensorflow as tf
import numpy as np
import tflearn

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, action_dim, low_act_bound, high_act_bound, learning_rate, tau, num_actor_vars, MINIBATCH_SIZE,LOGPATH):
        self.sess = sess
        self.LOGPATH = LOGPATH
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.otheraction, self.out, self.model = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_otheraction, self.target_out, self.target_model = self.create_critic_network()

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
        choice_grad = tf.slice(n2, [0,0,0], [-1, -1, 4])
        params_grad = tf.slice(n2, [0,0,4], [-1, -1, 6])
        params = tf.reshape(tf.slice(self.action, [0,4], [-1, 6]), [1, -1, 6])


        high = tf.constant([[high_act_bound.tolist()]*MINIBATCH_SIZE])
        # print high
        # print params
        low = tf.constant([[low_act_bound.tolist()]*MINIBATCH_SIZE])
        # print high - params
        pmax = tf.div((high - params), (high - low))
        pmin = tf.div((params - low), (high - low))
        # print pmax

        comparison = tf.less(tf.constant(0.0), params_grad)

        # print comparison
        # print comparison
        # print pmax
        self.action_grads =  tflearn.merge([choice_grad, tf.multiply(params_grad, tf.where(comparison, pmax, pmin))], axis = 2, mode ='concat')
        # print self.action_grads


    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])

        otheraction = tflearn.input_data(shape=[None, 2 * self.a_dim])

        in1 = tflearn.merge([inputs, action, otheraction], 'concat')

        l = tflearn.fully_connected(in1, 1024)
        la = tflearn.activations.leaky_relu(l, alpha=-.01)
        l2 = tflearn.fully_connected(la, 512)
        l2a = tflearn.activations.leaky_relu(l2, alpha=-.01)
        l3 = tflearn.fully_connected(l2a, 256)
        l3a = tflearn.activations.leaky_relu(l3, alpha=-.01)
        l4 = tflearn.fully_connected(l3a, 128)
        l4a = tflearn.activations.leaky_relu(l4, alpha=-.01)
        w_init = tflearn.initializations.normal(stddev = -0.01)
        out = tflearn.fully_connected(l4a, 1, weights_init=w_init)

        model = tflearn.DNN(out, session = self.sess)

        return inputs, action, otheraction, out, model

    def model_save(self, saveto, target):
        if target:
            self.target_model.save(saveto)
        else:
            self.model.save(saveto)


    def model_load(self, loadfrom, target):
        if target:
            self.target_model.load(loadfrom, weights_only=True, create_new_session=False)
        else:
            self.model.load(loadfrom, weights_only=True, create_new_session=False)

    def train(self, inputs, action, otheraction, predicted_q_value):
        return self.sess.run([self.out, self.loss, self.optimize], feed_dict={
        # return self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.otheraction: otheraction,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action, otheraction):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.otheraction: otheraction
        })

    def predict_target(self, inputs, action, otheraction):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.otheraction: otheraction
        })

    def action_gradients(self, inputs, actions, otheraction):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions,
            self.otheraction: otheraction
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
