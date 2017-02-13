#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

import sys, itertools
from hfo import *
import numpy as np
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from ddpf import ReplayBuffer
import ddpg
import actor_hfo
import critic_hfo

gamma = 0.3

def main():
    # Create the HFO Environment
    hfo = HFOEnvironment()
    # Connect to the server with the specified
    # feature set. See feature sets in hfo.py/hfo.hpp.
    hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,
                        'bin/teams/base/config/formations-dt', 6000,
                        'localhost', 'base_left', False)

    critic_network = CriticNetworkModel()
    target_critic_network = CriticNetworkModel()
    target_critic_network.initialize_equal(critic_network)

    actor_network = ActorNetworkModel()
    target_actor_network = ActorNetworkModel()
    target_actor_network.initialize_equal(actor_network)

    replay_buffer = ReplayBuffer()

    for episode in itertools.count():

        status = IN_GAME
        # Grab the state features from the environment
        s1 = hfo.getState()
        while status == IN_GAME:

            # # Grab the state features from the environment
            # features = hfo.getState()
            s = s1

            # get action
            action = actor_network.select_action(s)

            # Make action and step forward in time
            hfo.act(action)
            status = hfo.step()

            # Get new state s_(t+1)
            s1 = hfo.getState()

            # If game has finished, calculate reward based on whether or not a goal was scored
            if s1 != IN_GAME:
                if status == 1:
                    reward = 99999
                elif status == 2:
                    reward = -99999

            # Else calculate reward as distance between ball and goal
            else:
                reward = np.sqrt((s1[3] - 1)**2 + (s1[4])**2)
            
            # Store transition
            replay_buffer.addToBuffer(s, action, reward, s1)

            # Get minibatch
            if replay_buffer.getSize() > 1000:
                mb = replay_buffer.getMinibatch(1000)

            # Calculate y_i values
            y = ddpg.calculate_y(mb, target_critic_network, target_actor_network, gamma)

            # Update critic and actor
            critic_network.updateNetwork(mb, y)
            actor_network.updateNetwork(mb, critic_network)

            # Update target networks
            target_critic_network.updateTarget(tau, critic_network)
            target_actor_network.updateTarget(tau, actor_network)




        # Check the outcome of the episode
        print('Episode %d ended with %s'%(episode, hfo.statusToString(status)))

        # Quit if the server goes down
        if status == SERVER_DOWN:
            hfo.act(QUIT)
            break

if __name__ == '__main__':
    main()
