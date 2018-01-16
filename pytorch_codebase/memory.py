from collections import namedtuple
import random
import numpy as np
Experience = namedtuple('Experience',
                        ('states', 'actions', 'next_states', 'rewards'))


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.sorted_memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

        if self.position % 1000 == 0:
            self.memory = sorted(
                self.memory, key=lambda row: (row.rewards).mean())

    def sample(self, batch_size, prioritized=False):
        if prioritized:
            batch1 = random.sample(
                self.memory[:(self.capacity / 10)], batch_size / 4)
            batch2 = random.sample(
                self.memory[-(self.capacity / 10):], batch_size / 4)
            batch3 = random.sample(self.memory, batch_size / 2)
            return batch3 + batch2 + batch1
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
