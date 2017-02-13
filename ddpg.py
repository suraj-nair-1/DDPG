# Implementation of running DDPG algorithm for reinforcement learning on
# continuous action spaces to be used by HFO agents.

import sys, itertools

def calculate_y(mb, target_critic_network, target_actor_network, gamma):
	y_is = []
	for transition in mb:
		policy_action = target_actor_network.get_action(transition[3])
		Q_val = target_critic_network.get_Q(transition[3], policy_action)
		y_is.append(transition[2] + gamma * Q_val)

	return y_is


class ReplayBuffer(object):

    self.buffer = None
    self.size = None

    def __init__(self):  # Initialize buffer
    	self.buffer = []
        self.size = 0

    def getMinibatch(self, n):
    	inds = np.random.randint(0, len(self.buffer), n)
    	batch = [self.buffer[i] for i in inds]
    	return batch

    def addToBuffer(self, st, at, rt, st1):
    	self.buffer.append([st, at, rt, st1])
        self.size += 1

    def getSize(self):
        return self.size
