# Actor network for DDPG

import numpy as np
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout

class ActorNetworkModel(object):

    self.model = None
    self.explorationNoise = 5

    def __init__(self):  # Initialize random
        actor = Sequential()
        actor.add(Dense(128, input_dim = 19)
        actor.add(Activation('relu'))
        actor.add(Dense(1))
        actor.compile(loss='mse',optimizer='adam', metrics=['accuracy'])

        self.model = critic

    def initialize_equal(otherActorNetwork):
        update self.model
        update self.explorationNoise

    def select_action(curr_state):
        # use explorationNoise
        return action + np.random.uniform(-self.explorationNoise, self.explorationNoise)

    def get_action(state):
        return action

    def updateNetwork(mb, critic_network):
        return

    def updateTarget(tau, actor_network):
        theta_mu = actor_network.getWeights()
        setWeights(tau * theta_mu + (1 - tau) * self.getWeights())

    def getWeights():
        return self.model.get_weights()

    def setWeights(new_theta):
        self.model.set_weights(new_theta)
