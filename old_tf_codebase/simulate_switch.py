#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

import sys, itertools
from hfo import *

import numpy as np
import tensorflow as tf
import tflearn
import time
from actor_replay import ActorNetworkReplay


# Max training steps
MAX_EPISODES = 500000
# Max episode length
MAX_EP_STEPS = 500
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = .001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = .001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.0001
LOGPATH = "../DDPG/"
RANDOM_SEED = 1234

GPUENABLED = False
ORACLE = False
def main(_):
    ITERATIONS = 0.0

    # Create the HFO Environment
    hfo = HFOEnvironment()
    # Connect to the server with the specified
    # feature set. See feature sets in hfo.py/hfo.hpp.
    hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
                        'bin/teams/base/config/formations-dt', int(sys.argv[1]),
                        'localhost', 'base_left', False)

    state_dim = 74
    action_dim = 10
    low_action_bound = np.array([0., -180., -180., -180., 0., -180.])
    high_action_bound = np.array([100., 180., 180., 180., 100., 180.])

    CURR_MODEL = int(sys.argv[2])
    PLAYER = int(sys.argv[2])


    if PLAYER == 1:
        OTHERPLAYER = 2
        actor = ActorNetworkReplay(state_dim, action_dim, LOGPATH + "models/" +  sys.argv[3])
    else:
        OTHERPLAYER = 1
        actor = ActorNetworkReplay(state_dim, action_dim, LOGPATH + "models/" +  sys.argv[4])

    for i in xrange(MAX_EPISODES):


        status = IN_GAME
        # Grab the state features from the environment
        # s1 = hfo.getState()

        for j in xrange(MAX_EP_STEPS):


            # # Grab the state features from the environment
            # features = hfo.getState()
            s = hfo.getState()
            s_noise = np.reshape(s, (1, state_dim))

            # print sys.argv[2], s[66:]

            curr_ball_prox = s[53]


            f = open(LOGPATH+'intermediate_switch'+str(PLAYER)+'.txt', 'w')
            f.write(str(curr_ball_prox))
            f.close()

            while True:
                try:
                    otherprox = np.loadtxt(LOGPATH + "intermediate_switch"+str(OTHERPLAYER)+".txt", delimiter=",")
                    if len(otherprox.shape) == 0:
                        break
                except:
                    continue

            if otherprox < curr_ball_prox:
                if CURR_MODEL == 2:
                    # print "PLAYER", PLAYER, "SWITCHING"
                    actor.model_load(LOGPATH + "models/" + sys.argv[3])
                    CURR_MODEL = 1
            else:
                if CURR_MODEL == 1:
                    # print "PLAYER", PLAYER, "SWITCHING"
                    actor.model_load(LOGPATH + "models/" + sys.argv[4])
                    CURR_MODEL = 2


            # Added exploration noise #+ np.random.rand(1, 58)
            a = actor.model_predict(s_noise)[0]
            # model_a = a\.predict(s_noise)[0]
            # print a
            index = np.argmax(a[:4])
            # model_index, model_a = model_actor.add_noise(model_a, 0)
            # print PLAYER, "******************************"
            if index == 0:
                action  = (DASH, a[4], a[5])
            elif index == 1:
                action = (TURN, a[6])
            elif index == 2:
                action = (TACKLE, a[7])
            else:
                action = (KICK, a[8], a[9])
            print PLAYER, action
            # Make action and step forward in time
            # print action
            hfo.act(*action)
            # print "\n"
            terminal = hfo.step()


if __name__ == '__main__':
    tf.app.run()
