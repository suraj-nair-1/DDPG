#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

import sys, itertools
from hfo import *

import numpy as np
import tensorflow as tf
import tflearn
import os
import traceback

import time

from replay_buffer_maddpg import ReplayBuffer
from actor_hfo import ActorNetwork
from critic_hfo_maddpg import CriticNetwork


LOGPATH = "../DDPG/"
# LOGPATH = "/cs/ml/ddpgHFO/DDPG/"

PRIORITIZED = True

# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 500
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = .001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = .001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# Noise for exploration
EPS_GREEDY_INIT = 1.0
# EPS_ITERATIONS_ANNEAL = 1000000

# sigma = 1.0
# sigma_ep_anneal = 2000
# Parameters in format of theta-mu-sigma
# OU_NOISE_PARAMS = [[5.0, 0.0, 3.0], [5.0, 0.0, 3.0], [5.0, 0.0, 3.0],
#                    [5.0, 0.0, 3.0], [5.0, 0.0, 3.0], [5.0, 0.0, 3.0]]

# OU_NOISE_PARAMS = [[.1, 0.0, 3.0]] * 6

# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
# Size of replay buffer
BUFFER_SIZE = 1000000
MINIBATCH_SIZE = 1024

GPUENABLED = False
ORACLE = False


def main(_):
    PORT, OFFENSE, PLAYER, RANDOM_SEED = map(int, sys.argv[1:])
    print PORT, OFFENSE, PLAYER, RANDOM_SEED

    LOGNUM = 79

    if GPUENABLED:
        device = "/gpu:0"
    else:
        device = "/cpu:0"
    with tf.device(device):
        with tf.Session() as sess:

            # Create the HFO Environment
            # print "A"
            hfo = HFOEnvironment()
            # print "AA"
            # Connect to the server with the specified
            # feature set. See feature sets in hfo.py/hfo.hpp.
            print "CONNECTING ..."
            if OFFENSE:
                EPS_ITERATIONS_ANNEAL = 1000000
                hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
                    'bin/teams/base/config/formations-dt', PORT,
                    'localhost', 'base_left', False)
            else:
                EPS_ITERATIONS_ANNEAL = 3000000
                hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
                    'bin/teams/base/config/formations-dt', PORT,
                    'localhost', 'base_right', True)

            print "CONNECTED"
            ITERATIONS = 0.0
            NUM_GOALS = 0.0
            CURR_MODEL = PLAYER

            # print "AAAA"

            np.random.seed(RANDOM_SEED)
            tf.set_random_seed(RANDOM_SEED)

            state_dim = 74
            action_dim = 10
            low_action_bound = np.array([0., -180., -180., -180., 0., -180.])
            high_action_bound = np.array([100., 180., 180., 180., 100., 180.])

            if OFFENSE:

                actor_closer = ActorNetwork(sess, state_dim, action_dim, low_action_bound, \
                    high_action_bound, ACTOR_LEARNING_RATE, TAU, LOGPATH, sys.argv[2])

                critic_closer = CriticNetwork(sess, state_dim, action_dim, low_action_bound, high_action_bound, \
                    CRITIC_LEARNING_RATE, TAU, actor_closer.get_num_trainable_vars(), MINIBATCH_SIZE, LOGPATH)

                actor_farther = ActorNetwork(sess, state_dim, action_dim, low_action_bound, \
                    high_action_bound, ACTOR_LEARNING_RATE, TAU, LOGPATH, sys.argv[2])

                critic_farther = CriticNetwork(sess, state_dim, action_dim, low_action_bound, high_action_bound, \
                    CRITIC_LEARNING_RATE, TAU, actor_farther.get_num_trainable_vars(), MINIBATCH_SIZE, LOGPATH)

                # actor = ActorNetwork(sess, state_dim, action_dim, low_action_bound, \
                #     high_action_bound, ACTOR_LEARNING_RATE, TAU, LOGPATH, PLAYER)

                # critic = CriticNetwork(sess, state_dim, action_dim, low_action_bound, high_action_bound, \
                #     CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), MINIBATCH_SIZE, LOGPATH)


                if PLAYER == 1:
                    OTHERPLAYER = 2
                    actor = actor_closer
                    critic = critic_closer
                else:
                    OTHERPLAYER = 1
                    actor = actor_farther
                    critic = critic_farther

                # Set up summary Ops
                sess.run(tf.global_variables_initializer())

                print "INITIALIZED VARIABLES"


                # Initialize target network weights
                actor_closer.update_target_network()
                critic_closer.update_target_network()
                # Initialize target network weights
                actor_farther.update_target_network()
                critic_farther.update_target_network()

                # Initialize replay memory
                replay_buffer_closer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
                replay_buffer_farther = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
                if PLAYER == 1:
                    replay_buffer = replay_buffer_closer
                else:
                    replay_buffer = replay_buffer_farther

            else:
                actor = ActorNetwork(sess, state_dim, action_dim, low_action_bound, \
                high_action_bound, ACTOR_LEARNING_RATE, TAU, LOGPATH, PLAYER)

                critic = CriticNetwork(sess, state_dim, action_dim, low_action_bound, high_action_bound, \
                    CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(), MINIBATCH_SIZE, LOGPATH)


                if PLAYER == 1:
                    OTHERPLAYER = 2
                else:
                    OTHERPLAYER = 1

                # Set up summary Ops
                sess.run(tf.global_variables_initializer())

                print "INITIALIZED VARIABLES"


                # Initialize target network weights
                actor.update_target_network()
                critic.update_target_network()

                # Initialize replay memory
                replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
            num_0_row = 0



            for i in xrange(MAX_EPISODES):

                try:

                    ep_reward = 0.0
                    ep_ave_max_q = 0.0
                    # OU_NOISE_PARAMS = [[.1, 0.0, max(0.0, sigma - float(i) / sigma_ep_anneal)]] * 6

                    status = IN_GAME
                    # Grab the state features from the environment
                    # s1 = np.concatenate((hfo.getState(), np.ones((8,))), axis =0)
                    s1 = hfo.getState()
                    old_reward = 0
                    critic_loss = 0.0

                    ep_good_q = 0.0
                    ep_bad_q = 0.0
                    ep_move_q = 0.0
                    ep_turn_q = 0.0
                    ep_tackle_q = 0.0
                    ep_kick_q = 0.0
                    ep_updates = 0.0
                    ep_val_r = 0.0
                    # print "********************"
                    # print "Episode", i
                    # print "********************"
                    ep_switches  = 0.0

                    for j in xrange(MAX_EP_STEPS):


                        # # Grab the state features from the environment
                        # features = hfo.getState()
                        s = s1

                        # Added exploration noise
                        s_noise = np.reshape(s, (1, state_dim)) #+ np.random.rand(1, 19)

                        assert(np.isnan(s_noise).any() == False)
                        # print s_noise
                        a = actor.predict(s_noise)[0]

                        index, a = actor.add_noise(a, max(0.1, EPS_GREEDY_INIT - ITERATIONS / EPS_ITERATIONS_ANNEAL))


                        if index == 0:
                            action  = (DASH, a[4], a[5])
                        elif index == 1:
                            action = (TURN, a[6])
                        elif index == 2:
                            action = (TACKLE, a[7])
                        else:
                            action = (KICK, a[8], a[9])

                        # Make action and step forward in time
                        # print action
                        hfo.act(*action)
                        # print "\n"
                        terminal = hfo.step()

                        # Get new state s_(t+1)
                        s1 = hfo.getState()

                        curr_ball_prox = s1[53]
                        curr_kickable = s1[12]

                        np.savetxt(LOGPATH+'actions'+str(LOGNUM)+'_'+str(OFFENSE) + "_"+str(PLAYER)+'.txt', a.flatten())


                        # print PLAYER, curr_ball_prox
                        while True:
                            try:
                                if OFFENSE:
                                    other1 = np.loadtxt(LOGPATH+'actions'+str(LOGNUM)+'_'+str(OFFENSE) + "_"+str(OTHERPLAYER)+'.txt')
                                    other2 = np.loadtxt(LOGPATH+'actions'+str(LOGNUM)+'_0_3.txt')
                                else:
                                    other1 = np.loadtxt(LOGPATH+'actions'+str(LOGNUM)+'_1_'+str(1)+'.txt')
                                    other2 = np.loadtxt(LOGPATH+'actions'+str(LOGNUM)+'_1_'+str(2)+'.txt')

                                other = np.concatenate([other1, other2], axis = 0)
                                assert(other.shape == (20,))
                                break
                            except:
                                print "PLAYER", PLAYER, "FETCH FAILED"
                                # print e
                                continue

                        # print other.shape
                        if OFFENSE:
                            send_data  = np.array([curr_ball_prox, curr_kickable])
                            np.savetxt(LOGPATH+'intermediate'+str(LOGNUM)+'_'+str(PLAYER)+'.txt', send_data.flatten())


                            # print PLAYER, curr_ball_prox
                            while True:
                                try:
                                    aaa= np.loadtxt(LOGPATH + "intermediate"+str(LOGNUM)+'_'+str(OTHERPLAYER)+".txt")
                                    if len(aaa) == 2:
                                        otherprox, otherkickable = aaa
                                        break
                                except:
                                    print "PLAYER", PLAYER, "FETCH FAILED"
                                    continue




                        goal_proximity = s1[15]
                        ball_dist = 1.0 - curr_ball_prox
                        goal_dist = 1.0 - goal_proximity
                        ball_ang_sin_rad = s1[51]
                        ball_ang_cos_rad = s1[52]
                        ball_ang_rad = np.arccos(ball_ang_cos_rad)
                        if ball_ang_sin_rad < 0:
                            ball_ang_rad *= -1.
                        goal_ang_sin_rad = s1[13]
                        goal_ang_cos_rad = s1[14]
                        goal_ang_rad = np.arccos(goal_ang_cos_rad)
                        if goal_ang_sin_rad < 0:
                            goal_ang_rad *= -1.
                        alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)
                        # Law of Cosines
                        curr_goal_dist = np.sqrt(ball_dist*ball_dist + goal_dist*goal_dist - 2.*ball_dist*goal_dist*np.cos(alpha))

                        # val = 2.0 * ((np.max([curr_ball_prox + 1, otherprox + 1]) / np.sum([curr_ball_prox + 1, otherprox + 1])) - 0.5)

                        # print curr_ball_prox
                        # print curr_goal_dist



                        r = 0.0
                        # print j
                        if j != 0:

                            if OFFENSE:

                                # If game has finished, calculate reward based on whether or not a goal was scored
                                if terminal != IN_GAME:
                                    if int(terminal) == 1:
                                        NUM_GOALS += 1
                                        r += 5
                                else:
                                    # Movement to ball
                                    r +=  (curr_ball_prox - old_ball_prox)

                                    # Seperation between players
                                    # r += 0.2 * (val - oldval)
                                    # ep_val_r += (val - oldval)

                                    r += -3.0 * float(curr_goal_dist - old_goal_dist)
                                    # print r
                                    # if (old_kickable == -1) and (curr_kickable == 1):
                                    #     r += 1
                                    # # if (old_other_kickable == -1) and (otherkickable == 1):
                                    #     r += 1

                            else:
                                if terminal != IN_GAME:
                                    if int(terminal) == 1:
                                        NUM_GOALS += 1
                                        r += -5
                                    elif int(terminal) == 2:
                                        r += 5

                                else:
                                    r +=  (curr_ball_prox - old_ball_prox)

                                    r += 3.0 * float(curr_goal_dist - old_goal_dist)
                                    # print r
                                    # if (old_kickable == -1) and (curr_kickable == 1):
                                    #     r += 1


                                # print r

                        # print "\n\n\n"
                        old_ball_prox = curr_ball_prox
                        old_goal_dist = curr_goal_dist
                        old_kickable = curr_kickable
                        if OFFENSE:
                            old_other_kickable = otherkickable
                            old_other_ball_prox = otherprox
                        # oldval = val

                        # if r == 0:
                        #     r = -1
                        # print "Current Reward", r
                        


                        replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), np.reshape(other, (2 *actor.a_dim,)), r, \
                            terminal, np.reshape(s1, (actor.s_dim,)))

                        if OFFENSE:
                            if otherprox < old_ball_prox:
                                if CURR_MODEL == 2:
                                    actor = actor_closer
                                    critic = critic_closer
                                    replay_buffer = replay_buffer_closer
                                    ep_switches += 1
                                    CURR_MODEL = 1
                                
                            else:
                                if CURR_MODEL == 1:
                                    actor = actor_farther
                                    critic = critic_farther
                                    replay_buffer = replay_buffer_farther
                                    ep_switches += 1
                                    CURR_MODEL = 2


                        # TRAINING STEP
                        ###########################################################
                        if (replay_buffer.size() > MINIBATCH_SIZE) and (ITERATIONS % 10 == 0):

                            if (not PRIORITIZED) or (ITERATIONS < 200000) or (NUM_GOALS > 20):
                                s_batch, a_batch, othera_batch, r_batch, t_batch, s1_batch = \
                                    replay_buffer.sample_batch(MINIBATCH_SIZE)
                            else:
                                s_batch, a_batch,  othera_batch,  r_batch, t_batch, s1_batch = \
                                    replay_buffer.sample_batch_prioritized(MINIBATCH_SIZE)

                            ep_updates += 1

                            # print actor.predict_target(s_batch).shape
                            good_batch = []
                            bad_batch = []
                            for elem in s_batch:
                                ball_angle_sin = elem[51]
                                ang = np.degrees(np.arcsin(ball_angle_sin))
                                if ang > 0:
                                    bad_ang = ang - 180
                                else:
                                    bad_ang = ang + 180
                                good_batch.append([1, 0, 0, 0, 10, ang, 0, 0, 0, 0])
                                bad_batch.append([1, 0, 0, 0, 10, bad_ang, 0, 0, 0, 0])

                            move_batch = []
                            turn_batch = []
                            tackle_batch = []
                            kick_batch = []
                            for elem in s_batch:
                                move_batch.append([1, 0, 0, 0,np.random.uniform(0, 100) , np.random.uniform(-180, 180), 0, 0, 0, 0])
                                turn_batch.append([0, 1, 0, 0, 0, 0, np.random.uniform(-180, 180), 0, 0, 0])
                                tackle_batch.append([0, 0, 1, 0, 0, 0, 0, np.random.uniform(-180, 180), 0, 0])
                                kick_batch.append([0, 0, 0, 1, 0, 0, 0, 0, np.random.uniform(0, 100) , np.random.uniform(-180, 180)])

                                
                            target_good = critic.predict_target(s_batch, np.array(good_batch), othera_batch)
                            target_bad = critic.predict_target(s_batch, np.array(bad_batch), othera_batch)
                            target_move = critic.predict_target(s_batch, np.array(move_batch), othera_batch)
                            target_turn = critic.predict_target(s_batch, np.array(turn_batch), othera_batch)
                            target_tackle = critic.predict_target(s_batch, np.array(tackle_batch), othera_batch)
                            target_kick = critic.predict_target(s_batch, np.array(kick_batch), othera_batch)

                            ep_good_q += np.mean(target_good)
                            ep_bad_q += np.mean(target_bad)
                            ep_move_q += np.mean(target_move)
                            ep_turn_q += np.mean(target_turn)
                            ep_tackle_q += np.mean(target_tackle)
                            ep_kick_q += np.mean(target_kick)

                            target_q = critic.predict_target(s1_batch, actor.predict_target(s1_batch), othera_batch)

                            y_i = []
                            for k in xrange(MINIBATCH_SIZE):
                                if t_batch[k] != IN_GAME:
                                    y_i.append(r_batch[k])
                                else:
                                    y_i.append(r_batch[k] + GAMMA * target_q[k])

                            # Update the critic given the targets
                            # print y_i
                            # print predicted_q_value
                            predicted_q_value, ep_critic_loss, _ = critic.train(s_batch, a_batch, othera_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
                            # predicted_q_value, ep_critic_loss = critic.getloss(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                            ep_ave_max_q += np.mean(predicted_q_value)
                            critic_loss += np.mean(ep_critic_loss)

                            # Update the actor policy using the sampled gradient
                            a_outs = actor.predict(s_batch)
                            grads = critic.action_gradients(s_batch, a_outs, othera_batch)
                            actor.train(s_batch, grads[0])

                            # Update target networks
                            actor.update_target_network()
                            critic.update_target_network()


                            if (ITERATIONS % 1000000) == 0:
                                    if OFFENSE:
                                        actor_closer.model_save(LOGPATH + "models/targetcloser_"+str(LOGNUM)+"_"+str(OFFENSE)+"_"+str(PLAYER)+"_"+str(ITERATIONS)+".tflearn", target=True)
                                        actor_farther.model_save(LOGPATH + "models/targetfarther_"+str(LOGNUM)+"_"+str(OFFENSE)+"_"+str(PLAYER)+"_"+str(ITERATIONS)+".tflearn", target=True)
                                    else:
                                        actor.model_save(LOGPATH + "models/target_"+str(LOGNUM)+"_"+str(OFFENSE)+"_"+str(PLAYER)+"_"+str(ITERATIONS)+".tflearn", target=True)
                            # break
                        ITERATIONS += 1
                        ep_reward += r

                        # EPISODE IS OVER
                        ###########################################################
                        if terminal:
                            print terminal

                            if j == 0:
                                num_0_row += 1
                            else:
                                num_0_row = 0

                            if num_0_row > 5:
                                print "HFO FAILING"
                                sys.exit()


                            f = open(LOGPATH +'logging/logs'+str(LOGNUM)+'_' + str(PLAYER) + '.txt', 'a')
                            f.write(str(float(ep_reward)) + "," + str(ep_ave_max_q / float(ep_updates+1))+ "," \
                                + str(float(critic_loss)/ float(ep_updates+1)) + "," +  \
                                str(EPS_GREEDY_INIT - ITERATIONS/ EPS_ITERATIONS_ANNEAL) + \
                                "," + str(ep_good_q / float(ep_updates+1)) + "," + str(ep_bad_q / float(ep_updates+1))\
                                + "," + str(ep_move_q / float(ep_updates+1)) + "," + str(ep_turn_q / float(ep_updates+1))\
                                + "," + str(ep_tackle_q / float(ep_updates+1)) + "," + str(ep_kick_q / float(ep_updates+1)) \
                                + "," + str(ep_switches) +"," +  str(ep_val_r)+ "," +  str(j)+ "\n")
                            f.close()

                            print('| Reward: ' , float(ep_reward), " | Episode", i, \
                                '| Qmax:',  (ep_ave_max_q / float(j+1)), ' | Critic Loss: ', float(critic_loss)/ float(j+1))

                            break



                except Exception as e:
                    print "EPISODE", i, "FAILED"
                    print str(e)
                    print traceback.print_exc()
                    sys.exit()
                    break

if __name__ == '__main__':
    tf.app.run()
