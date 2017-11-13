import sys, itertools
from hfo import *

import numpy as np
import tensorflow as tf
import threading
import torch
from torch.autograd import Variable
from multiprocessing import Pool
import multiprocessing
from MADDPG import MADDPG
import numpy as np
import torch as th
import time
# LOGPATH = "/cs/ml/ddpgHFO/DDPG/"
LOGPATH = "Users/surajnair/Documents/Tech/research/MADDPH_HFO"
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

# Size of replay buffer
capacity = 1000000
batch_size = 1024
episodes_before_train = 1

GPUENABLED = False
ORACLE = False
PORT = 4500

FloatTensor = torch.cuda.FloatTensor if GPUENABLED else torch.FloatTensor

def connect():
    hfo = HFOEnvironment()
    hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
                        'bin/teams/base/config/formations-dt', PORT,
                        'localhost', 'base_left', False)
    return hfo


def take_action_and_step(a, env):
    index = np.argmax(a[:4])
    if index == 0:
        action  = (DASH, a[4], a[5])
    elif index == 1:
        action = (TURN, a[6])
    elif index == 2:
        action = (TACKLE, a[7])
    else:
        action = (KICK, a[8], a[9])

    env.act(*action)
    terminal = env.step()
    s1 = env.getState()
    return s1, terminal


def get_curr_state_vars(s1):
    curr_ball_prox = s1[53]
    curr_kickable = s1[12]
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

    return curr_ball_prox, curr_goal_dist, curr_kickable


def get_rewards(terminal, curr_ball_prox, curr_goal_dist, curr_kickable,
    old_ball_prox, old_goal_dist, old_kickable):

    r = 0.0

    # If game has finished, calculate reward based on whether or not a goal was scored
    goal_made = False
    if terminal != IN_GAME:
        if int(terminal) == 1:
            goal_made = True
            r += 5
    else:
        # Movement to ball
        r +=  (curr_ball_prox - old_ball_prox)
        r += -3.0 * float(curr_goal_dist - old_goal_dist)

    return r, goal_made

def run_process(maddpg, player_num, player_queue):
    env = connect()

    ITERATIONS = 0.0
    NUM_GOALS = 0.0
    np.random.seed(2)
    for ep in xrange(MAX_EPISODES):
        ep_reward = 0.0
        ep_ave_max_q = 0.0

        status = IN_GAME
        states1 = env.getState()

        old_reward = 0
        critic_loss = 0.0
        ep_good_q = 0.0
        ep_bad_q = 0.0
        ep_move_q = 0.0
        ep_turn_q = 0.0
        ep_tackle_q = 0.0
        ep_kick_q = 0.0
        ep_updates = 0.0


        rr = np.zeros((1,))
        old_ball_proxs = np.zeros((1,))
        old_goal_dists = np.zeros((1,))
        old_kickables = np.zeros((1,))
        for j in xrange(MAX_EP_STEPS):
            # # Grab the state features from the environment
            states = states1
            try:
                states =  torch.from_numpy(states).float()
            except:
                states = states.float()
            states = Variable(states).type(FloatTensor)

            actions = maddpg.select_action(states, player_num).data.cpu()
            states1, terminal = take_action_and_step(actions, env)


            curr_ball_proxs, curr_goal_dists, curr_kickables = get_curr_state_vars(states1)

            action_rewards = np.zeros((1,))
            if j != 0:
                # print curr_ball_proxs,curr_goal_dists
                action_rewards, goal_made = get_rewards(terminal, curr_ball_proxs, curr_goal_dists, curr_kickables, old_ball_proxs, old_goal_dists, old_kickables)


                if np.any(goal_made):
                    NUM_GOALS += 1

            rr += action_rewards
            old_ball_proxs = curr_ball_proxs
            old_goal_dists = curr_goal_dists
            old_kickables = curr_kickables

            states1 = np.stack(states1)
            states1 =torch.from_numpy(states1).float()


            # TODO Anshul/Suraj: Start updating this so each process (agent) pushes to memory
            # then after both are pushed they are aligned and saved like normal
            # Will require changing the memory.py class
            if j == MAX_EP_STEPS - 1:
                player_queue.put((states.data, actions, None, rr))
            else:
                player_queue.put((states.data, actions, states1, rr))
            states = states1


            # EPISODE IS OVER
            ###########################################################
            if terminal:
                print terminal
                print('| Reward: ' , rr, " | Episode", ep)


            ################################
            # TODO Anshul: Modify the logging to log info for all k agents.

            # TODO Anshul: Modify the DDPG_Performance IPYNB to plot from the new
            # logging format
            ################################
                # f = open(LOGPATH +'logging/logs'+str(LOGNUM)+'_' + str(PLAYER) + '.txt', 'a')
                # f.write(str(float(ep_reward)) + "," + str(ep_ave_max_q / float(ep_updates+1))+ "," \
                #     + str(float(critic_loss)/ float(ep_updates+1)) + "," +  \
                #     str(EPS_GREEDY_INIT - ITERATIONS/ EPS_ITERATIONS_ANNEAL) + \
                #     "," + str(ep_good_q / float(ep_updates+1)) + "," + str(ep_bad_q / float(ep_updates+1))\
                #     + "," + str(ep_move_q / float(ep_updates+1)) + "," + str(ep_turn_q / float(ep_updates+1))\
                #     + "," + str(ep_tackle_q / float(ep_updates+1)) + "," + str(ep_kick_q / float(ep_updates+1)) \
                #     + "," + str(ep_switches) +"," +  str(ep_val_r)+ "," +  str(j)+ "\n")
                # f.close()

                # print('| Reward: ' , float(ep_reward), " | Episode", i, \
                #     '| Qmax:',  (ep_ave_max_q / float(j+1)), ' | Critic Loss: ', float(critic_loss)/ float(j+1))

                break



def run():
    n_agents= 2
    n_states = 77
    n_actions = 10

    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()

    maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,episodes_before_train)

    p1 = multiprocessing.Process(target=run_process, args=(maddpg, 0, q1))
    p2 = multiprocessing.Process(target=run_process, args=(maddpg, 1, q2))

    print "Started"

    p1.start()
    time.sleep(5)
    p2.start()

    while True:
        p1_sts, p1_acts, p1_sts1, p1_rws = q1.get()
        p2_sts, p2_acts, p2_sts1, p2_rws = q2.get()

        sts = torch.stack([p1_sts, p2_sts])
        acts = torch.stack([p1_acts, p2_acts])
        if (p2_sts1 is None) or (p1_sts1 is None):
            sts1 = None
        else:
            sts1 = torch.stack([p1_sts1, p2_sts1])
        rws = np.stack([p1_rws, p2_rws])
        rws = torch.FloatTensor(rws)

        maddpg.memory.push(sts, acts, sts1, rws)
        c_loss, a_loss = maddpg.update_policy()


if __name__ == '__main__':
    run()


















