import sys, itertools
from hfo import *

import numpy as np
import tensorflow as tf
import torch
from torch.autograd import Variable
from multiprocessing import Pool
from MADDPG import MADDPG
import numpy as np
import torch as th
import time
LOGPATH = "/cs/ml/ddpgHFO/DDPG/"

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

def connect(slp):
    time.sleep(slp)
    hfo = HFOEnvironment()
    hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
                        'bin/teams/base/config/formations-dt', PORT,
                        'localhost', 'base_left', False)
    print "FINISHED"
    return hfo

def get_states(env):
    print "GETTING STATE OG"
    # time.sleep(np.)
    s = env.getState()
    return s

def take_action_and_step(inpt):
    a, env = inpt
    index = np.argmax(a[:4])
    if index == 0:
        action  = (DASH, a[4], a[5])
    elif index == 1:
        action = (TURN, a[6])
    elif index == 2:
        action = (TACKLE, a[7])
    else:
        action = (KICK, a[8], a[9])
    print "STEPPING"
    env.act(*action)
    terminal = env.step()
    print "STEP FINISHED"
    s1 = env.getState()
    return s1




def run():
    n_agents= int(sys.argv[1])
    n_states = 77
    n_actions = 10

    # agent_envs = []
    p = Pool(5)


    agent_envs = p.map(connect, [0, 5])
    # for k in range(n_agents):
    #     print k, PORT
    #     hfo = HFOEnvironment()
    #     hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
    #                     'bin/teams/base/config/formations-dt', PORT,
    #                     'localhost', 'base_left', False)
    #     agent_envs.append(hfo)
    #     print "DONE"

    maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,episodes_before_train)
    FloatTensor = torch.cuda.FloatTensor if maddpg.use_cuda else torch.FloatTensor

    ITERATIONS = 0.0
    NUM_GOALS = 0.0
    np.random.seed(2)
    print "STARTING EPISODES"
    for ep in xrange(MAX_EPISODES):
        ep_reward = 0.0
        ep_ave_max_q = 0.0

        status = IN_GAME
        # Grab the state features from the environment
        # s1 = np.concatenate((hfo.getState(), np.ones((8,))), axis =0)''
        states1 = p.map(get_states, agent_envs)
        # print "GOTEEM"
        states1 = np.stack(states1)
        print states1.shape

        old_reward = 0
        critic_loss = 0.0

        ep_good_q = 0.0
        ep_bad_q = 0.0
        ep_move_q = 0.0
        ep_turn_q = 0.0
        ep_tackle_q = 0.0
        ep_kick_q = 0.0
        ep_updates = 0.0


        rr = np.zeros((n_agents,))
        old_ball_proxs = np.zeros((n_agents,))
        old_goal_dists = np.zeros((n_agents,))
        old_kickables = np.zeros((n_agents,))
        for j in xrange(MAX_EP_STEPS):
            # # Grab the state features from the environment
            states = states1
            states =  torch.from_numpy(states).float()
            states = Variable(states).type(FloatTensor)

            actions = maddpg.select_action(states).data.cpu()
            print actions.shape

            inpt = zip(actions, agent_envs)
            states1 = p.map(take_action_and_step, inpt)
            print states1.shape
            assert False
            # action = add_noise
            states1 = []
            rewards = []
            ################################
            # TODO Anshul: Switch this for loop 
            # to a function and use map
            ################################
            for i in range(actions.size()[0]):
                env = agent_envs[i]
                a = actions[i]
                index = np.argmax(a[:4])
                if index == 0:
                    action  = (DASH, a[4], a[5])
                elif index == 1:
                    action = (TURN, a[6])
                elif index == 2:
                    action = (TACKLE, a[7])
                else:
                    action = (KICK, a[8], a[9])
                print "STEPPING"
                env.act(*action)
                terminal = env.step()
                print "STEP FINISHED"
                s1 = env.getState()
                states1.append(s1)


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

                r = 0.0
                # print j
                if j != 0:
                    # If game has finished, calculate reward based on whether or not a goal was scored
                    if terminal != IN_GAME:
                        if int(terminal) == 1:
                            NUM_GOALS += 1
                            r += 5
                    else:
                        # Movement to ball
                        r +=  (curr_ball_prox - old_ball_proxs[i])
                        r += -3.0 * float(curr_goal_dist - old_goal_dists[i])
                rr[i] += r
                old_ball_proxs[i] = curr_ball_prox
                old_goal_dists[i] = curr_goal_dist
                old_kickables[i] = curr_kickable
            rewards.append(r)
            rewards = np.array(rewards)
            rewards = torch.FloatTensor(rewards).type(FloatTensor)
            states1 = np.stack(states1)
            states1 =torch.from_numpy(states1).float()
            ################################################################


            if j == MAX_EP_STEPS - 1:
                maddpg.memory.push(states.data, actions, None, rewards)
            else:
                maddpg.memory.push(states.data, actions, states1, rewards)
            states = states1
            c_loss, a_loss = maddpg.update_policy()


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

if __name__ == '__main__':
    run()

