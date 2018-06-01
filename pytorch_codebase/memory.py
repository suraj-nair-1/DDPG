from collections import namedtuple
import random
import numpy as np

Experience = namedtuple(
    'Experience', ('states', 'actions', 'next_states', 'rewards'))
ExperienceOptions = namedtuple(
    'Experience', ('states', 'actions', 'next_states', 'rewards', 'option'))


class ReplayMemory:
    '''
    Replay Memory Buffer
    '''

    def __init__(self, capacity, option=False):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.option = option
        self.tkm = {'states': 0, 'actions': 1,
                    'next_states': 2, 'rewards': 3, 'option': 4}

    def push(self, *args):
        '''
        Push an experience to the buffer
        '''
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        item = args
        self.memory[self.position] = item

        self.position = (self.position + 1) % self.capacity

        if self.position % 1000 == 0:
            self.memory = sorted(
                self.memory, key=lambda row: (row[self.tkm["rewards"]]).mean())

    def sample(self, batch_size, prioritized=False):
        '''
        Draws a sample of size batch_size from the buffer
        '''
        if prioritized:
            batch1 = random.sample(
                self.memory[:int(self.capacity / 10)], int(batch_size / 4))
            batch2 = random.sample(
                self.memory[-int(self.capacity / 10):], int(batch_size / 4))
            batch3 = random.sample(self.memory, int(batch_size / 2))
            return batch3 + batch2 + batch1
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
