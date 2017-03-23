# Implementation of running DDPG algorithm for reinforcement learning on
# continuous action spaces to be used by HFO agents.

from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.sortedbuffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.sortedbuffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            # self.sortedbuffer.pop()
            self.sortedbuffer.append(experience)

    def size(self):
        return self.count

    def sample_batch_prioritized(self, batch_size):
        batch = []

        # while len(batch) < batch_size:
        worst = self.sortedbuffer[:(self.buffer_size / 10)]
        best = self.sortedbuffer[-(self.buffer_size / 10):]
        if np.random.uniform() < 0.01:
            self.sortedbuffer = sorted(self.sortedbuffer, key=lambda row: np.abs(row[2]))
            self.sortedbuffer = deque(worst + best)


        batch1 = random.sample(worst, batch_size / 4)
        batch2 = random.sample(best, batch_size / 4)
        batch3 = random.sample(self.buffer, batch_size / 2)
        batch = batch1 + batch2 + batch3

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def sample_batch(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.deque.clear()
        self.count = 0
