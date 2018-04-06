from collections import namedtuple
import random
import numpy as np

Experience = namedtuple(
    'Experience', ('states', 'actions', 'next_states', 'rewards'))
ExperienceOptions = namedtuple(
    'Experience', ('states', 'actions', 'next_states', 'rewards', 'option'))
    
class ReplayMemory:

    def __init__(self, capacity, option=False):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.option = option
        self.tkm = {'states': 0, 'actions': 1,
                    'next_states': 2, 'rewards': 3, 'option': 4}
        # self.Experience = 0

        if not option:
            pass
            # self.Experience = namedtuple('Experience',
            #                              ('states', 'actions', 'next_states', 'rewards'))
        else:
            # self.Experience = namedtuple('Experience',
            #                              ('states', 'actions', 'next_states', 'rewards', 'option'))
            self.option_mem = {0: {}, 1: {}}

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # print(args)
        if self.option:
            item = args
            # item = {'states': args[0], 'actions': args[1],
            #         'next_states': args[2], 'rewards': args[3], 'option': args[4]}
        else:
            item = args
            # item = {'states': args[0], 'actions': args[1],
            #         'next_states': args[2], 'rewards': args[3]}
        # item = self.Experience(*args)
        self.memory[self.position] = item

        if self.option:
            for player in range(2):
                p_opt = int(
                    np.argmax(item[self.tkm["option"]][player].cpu().numpy()))
                if p_opt in self.option_mem[player]:
                    self.option_mem[player][p_opt].append(item)
                else:
                    self.option_mem[player][p_opt] = [item]

                if len(self.option_mem[player][p_opt]) > self.capacity:
                    self.option_mem[player][p_opt] = self.option_mem[
                        player][p_opt][-self.capacity:]

        self.position = (self.position + 1) % self.capacity

        if self.position % 1000 == 0:
            self.memory = sorted(
                self.memory, key=lambda row: (row[self.tkm["rewards"]]).mean())
            if self.option:
                for p in range(2):
                    for key in self.option_mem[p].keys():
                        self.option_mem[p][key] = sorted(
                            self.option_mem[p][key], key=lambda row: (row[self.tkm["rewards"]]).mean())

    def sample(self, batch_size, prioritized=False):
        if prioritized:
            batch1 = random.sample(
                self.memory[:int(self.capacity / 10)], int(batch_size / 4))
            batch2 = random.sample(
                self.memory[-int(self.capacity / 10):], int(batch_size / 4))
            batch3 = random.sample(self.memory, int(batch_size / 2))
            return batch3 + batch2 + batch1
        return random.sample(self.memory, batch_size)

    def sample_option(self, batch_size, player,  option, prioritized=False):
        new_mem = self.option_mem[player][option]
        if prioritized:
            batch1 = random.sample(
                new_mem[:int(self.capacity / 10)], int(batch_size / 4))
            batch2 = random.sample(
                new_mem[-int(self.capacity / 10):], int(batch_size / 4))
            batch3 = random.sample(new_mem, int(batch_size / 2))
            return batch3 + batch2 + batch1
        return random.sample(new_mem, batch_size)

    def __len__(self):
        return len(self.memory)
