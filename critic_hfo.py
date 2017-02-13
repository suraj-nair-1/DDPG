# Critic network for DDPG

import numpy as np
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout

class CriticNetworkModel(object):

    self.model= None
    self.BATCH_SIZE = 32
    self.NB_EPOCH = 10

    def __init__(self):  # Initialize random
        critic = Sequential()
        critic.add(Dense(128, input_dim = 20)
        critic.add(Activation('relu'))
        critic.add(Dense(1))
        critic.compile(loss='mse',optimizer='adam', metrics=['accuracy'])

        self.model = critic

    def initialize_equal(self, otherCriticNetwork):
        otherCriticNetwork.setWeights(self.model.get_weights())
        

    def get_Q(self, state, action):
        input_feature = state + action
        self.model.predict(input_feature)

    def updateNetwork(self, mb, y):
        self.model.fit(mb, y, batch_size = self.BATCH_SIZE, nb_epoch = self.NB_EPOCH)

    def updateTarget(self, tau, critic_network):
        theta_Q = critic_network.getWeights()
        setWeights(tau * theta_Q + (1 - tau) * self.getWeights())

    def getWeights(self):
        return self.model.get_weights()

    def setWeights(self, new_theta):
        self.model.set_weights(new_theta)
