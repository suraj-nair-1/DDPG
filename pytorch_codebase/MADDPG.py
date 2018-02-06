from model import Critic, Actor, MetaCritic
import torch as th
from copy import deepcopy
from memory import ReplayMemory
from torch.optim import Adam
from randomProcess import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from params import scale_reward
import time
import os
import time


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class OMADDPG:

    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train, n_options):
        self.actors = [Actor(dim_obs, dim_act)
                       for i in range(n_agents * n_options)]
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents * n_options)]
        self.meta_critic = MetaCritic(n_agents, dim_obs, n_options)

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        self.meta_critics_target = deepcopy(self.meta_critic)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.n_options = n_options

        self.memory = ReplayMemory(capacity, option=True)
        self.batch_size = batch_size
        self.use_cuda = False  # th.cuda.is_available()

        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.meta_optimizer = Adam(self.meta_critic.parameters(), lr=0.0001)
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        if self.use_cuda:
            self.meta_critic.cuda()
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def save(self, logpath, lognum):
        path = logpath + "saved_models/run_" + \
            str(lognum) + "_" + str(self.episode_done)
        if not os.path.isdir(path):
            os.mkdir(path)
        for c, x in enumerate(self.actors_target):
            th.save(x, os.path.join(path, 'actor_agent_%d.pt' % (c)))
        for c, x in enumerate(self.critics_target):
            th.save(x, os.path.join(path, 'critic_agent_%d.pt' % (c)))

    def load(self, logpath, lognum, episode_done):
        path = logpath + "saved_models/run_" + \
            str(lognum) + "_" + str(episode_done)
        for i in range(self.n_agents):
            actor_path = os.path.join(path, 'actor_agent_%d.pt' % (i))
            critic_path = os.path.join(path, 'critic_agent_%d.pt' % (i))
            if os.path.exists(actor_path):
                self.actors[i] = th.load(actor_path)
                self.actors_target[i] = th.load(actor_path)
                self.critics[i] = th.load(critic_path)
                self.critics_target[i] = th.load(critic_path)

    def to_gpu(self):
        self.use_cuda = True
        self.meta_critic.cuda()
        for x in self.actors:
            x.cuda()
            x.low_action_bound.cuda()
            x.high_action_bound.cuda()
        for x in self.critics:
            x.cuda()
        for x in self.actors_target:
            x.cuda()
            x.low_action_bound.cuda()
            x.high_action_bound.cuda()
        for x in self.critics_target:
            x.cuda()

    def to_cpu(self):
        self.use_cuda = False
        self.meta_critic.cpu()
        for x in self.actors:
            x.cpu()
            x.low_action_bound.cpu()
            x.high_action_bound.cpu()
        for x in self.critics:
            x.cpu()
        for x in self.actors_target:
            x.cpu()
            x.low_action_bound.cpu()
            x.high_action_bound.cpu()
        for x in self.critics_target:
            x.cpu()

    def update_policy(self, prioritized=False):
        # do not train until exploration is enough
        # print 'update'
        t0 = time.time()
        if self.episode_done <= self.episodes_before_train:
            print "UPDATETIME", time.time() - t0
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(
                self.batch_size, prioritized=prioritized)
            batch = self.memory.Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = Variable(
                th.stack(batch.states).type(FloatTensor))
            action_batch = Variable(
                th.stack(batch.actions).type(FloatTensor))
            reward_batch = Variable(
                th.stack(batch.rewards).type(FloatTensor))
            option_batch = Variable(
                th.stack(batch.option).type(FloatTensor))

            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = Variable(th.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor))

            self.meta_optimizer.zero_grad()

            # for current agent
            meta_state = state_batch[:, agent]
            meta_option = option_batch[:, agent]
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            whole_option = option_batch.view(self.batch_size, -1)

            current_Q = self.meta_critic(meta_state, meta_option)

            directq = []
            for item in range(len(option_batch)):
                opt = option_batch[item, agent, :]

                opt = np.argmax(opt.cpu().data.numpy())

                q1 = self.critics_target[opt](
                    whole_state[item].unsqueeze(0), whole_action[item].unsqueeze(0))

                for o1 in range(self.n_options):
                    oo1 = Variable(
                        th.zeros(self.n_options).type(FloatTensor))
                    oo1[o1] = 1.0

                    td1 = self.meta_critic(non_final_next_states[item, agent, :].unsqueeze(
                        0), oo1.view(-1, self.n_options))
                    try:
                        if td1 > mx:
                            mx = td1
                    except:
                        mx = td1

                directq.append((q1 + self.GAMMA * mx).view(-1))

            directq = th.stack(directq)
            loss_Q = nn.MSELoss()(current_Q, directq.detach())
            loss_Q.backward()
            self.meta_optimizer.step()

            for opt in range(self.n_options):
                transitions = self.memory.sample_option(
                    self.batch_size, agent, opt, prioritized=prioritized)
                batch = self.memory.Experience(*zip(*transitions))
                non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                     batch.next_states)))
                # state_batch: batch_size x n_agents x dim_obs
                state_batch = Variable(
                    th.stack(batch.states).type(FloatTensor))
                action_batch = Variable(
                    th.stack(batch.actions).type(FloatTensor))
                reward_batch = Variable(
                    th.stack(batch.rewards).type(FloatTensor))
                option_batch = Variable(
                    th.stack(batch.option).type(FloatTensor))

                # : (batch_size_non_final) x n_agents x dim_obs
                non_final_next_states = Variable(th.stack(
                    [s for s in batch.next_states
                     if s is not None]).type(FloatTensor))

                # for current agent
                whole_state = state_batch.view(self.batch_size, -1)
                whole_action = action_batch.view(self.batch_size, -1)
                self.critic_optimizer[agent * self.n_options + opt].zero_grad()
                current_Q = self.critics[
                    agent * self.n_options + opt](whole_state, whole_action)

                non_final_next_actions = [
                    self.actors_target[i](non_final_next_states[:,
                                                                i,
                                                                :]) for i in range(
                                                                    self.n_agents)]
                non_final_next_actions = th.stack(non_final_next_actions)
    #            non_final_next_actions = Variable(non_final_next_actions)
                non_final_next_actions = (
                    non_final_next_actions.transpose(0,
                                                     1).contiguous())

                target_Q = Variable(th.zeros(
                    self.batch_size).type(FloatTensor))
                target_Q[non_final_mask] = self.critics_target[agent * self.n_options + opt](
                    non_final_next_states.view(-1,
                                               self.n_agents * self.n_states),
                    non_final_next_actions.view(-1,
                                                self.n_agents * self.n_actions))

                # scale_reward: to scale reward in Q functions
                target_Q = (target_Q * self.GAMMA) + (
                    reward_batch[:, agent] * scale_reward).view(-1)

                loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
                loss_Q.backward()
                self.critic_optimizer[agent * self.n_options + opt].step()

                self.actor_optimizer[agent * self.n_options + opt].zero_grad()
                state_i = state_batch[:, agent, :]
                action_i = self.actors[agent * self.n_options + opt](state_i)
                action_i.retain_grad()
                ac = action_batch.clone()
                ac[:, agent, :] = action_i
                whole_action = ac.view(self.batch_size, -1)
                actor_loss = - \
                    self.critics[agent * self.n_options +
                                 opt](whole_state, whole_action)
                actor_loss = actor_loss.mean()
                actor_loss.backward()

                params = action_i[:, 4:]
                high = self.actors[
                    agent * self.n_options + opt].high_action_bound
                high = high.repeat(action_i.size()[0], 1)
                low = self.actors[
                    agent * self.n_options + opt].low_action_bound
                low = low.repeat(action_i.size()[0], 1)

                if params.data.type() == 'torch.cuda.FloatTensor':
                    high = high.cuda()
                    low = low.cuda()

                pmax = ((high - params) / (high - low))
                pmin = ((params - low) / (high - low))

                grad = action_i.grad[:, 4:]
                g1 = (grad < 0).float() * pmin
                g2 = (grad >= 0).float() * pmax

                action_i.grad = th.cat([action_i.grad[:, :4], (g1 + g2)], 1)

                self.actor_optimizer[agent * self.n_options + opt].step()
                c_loss.append(loss_Q)
                a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(len(self.actors)):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        print "UPDATETIME", time.time() - t0
        # print c_loss, a_loss
        return c_loss, a_loss

    def select_action(self, state_batch, i):
        for o1 in range(self.n_options):
            oo1 = Variable(th.zeros(self.n_options).type(th.FloatTensor))
            oo1[o1] = 1.0
            td1 = self.meta_critic(state_batch.view(
                1, -1), oo1.view(-1, self.n_options))
            try:
                if td1 > mx:
                    mx = td1
                    mx_o = o1
            except:
                mx = td1
                mx_o = o1

        act = self.actors[i * self.n_options + o1](state_batch.view(1, -1))
        # self.steps_done += 1
        return act[0], o1

    def critic_predict(self, state_batch, action_batch, i, opt):
        # for o1 in range(self.n_options):
        #     for o2 in range(self.n_options):
        #         oo1 = Variable(th.zeros(self.n_options).type(th.FloatTensor))
        #         oo1[o1] = 1.0
        #         oo2 = Variable(th.zeros(self.n_options).type(th.FloatTensor))
        #         oo2[o2] = 1.0

        #         print state_batch.size()
        #         print th.stack([oo1, oo2]).view(-1, self.n_agents * self.n_options).size()
        #         td1 = self.meta_critic(state_batch[i].view(1, -1), th.stack([oo1, oo2]).view(-1,
        #                                                                                      self.n_agents * self.n_options))
        #         try:
        #             if td1 > mx:
        #                 mx = td1
        #                 mx_o = [o1, o2]
        #         except:
        #             mx = td1
        #             mx_o = [o1, o2]

        return self.critics[i * self.n_options + opt](state_batch, action_batch)


class MADDPG:

    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = False  # th.cuda.is_available()

        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def save(self, logpath, lognum):
        path = logpath + "saved_models/run_" + \
            str(lognum) + "_" + str(self.episode_done)
        if not os.path.isdir(path):
            os.mkdir(path)
        for c, x in enumerate(self.actors_target):
            th.save(x, os.path.join(path, 'actor_agent_%d.pt' % (c)))
        for c, x in enumerate(self.critics_target):
            th.save(x, os.path.join(path, 'critic_agent_%d.pt' % (c)))

    def load(self, logpath, lognum, episode_done):
        path = logpath + "saved_models/run_" + \
            str(lognum) + "_" + str(episode_done)
        for i in range(self.n_agents):
            actor_path = os.path.join(path, 'actor_agent_%d.pt' % (i))
            critic_path = os.path.join(path, 'critic_agent_%d.pt' % (i))
            if os.path.exists(actor_path):
                self.actors[i] = th.load(actor_path)
                self.actors_target[i] = th.load(actor_path)
                self.critics[i] = th.load(critic_path)
                self.critics_target[i] = th.load(critic_path)

    def to_gpu(self):
        self.use_cuda = True
        for x in self.actors:
            x.cuda()
            x.low_action_bound.cuda()
            x.high_action_bound.cuda()
        for x in self.critics:
            x.cuda()
        for x in self.actors_target:
            x.cuda()
            x.low_action_bound.cuda()
            x.high_action_bound.cuda()
        for x in self.critics_target:
            x.cuda()

    def to_cpu(self):
        self.use_cuda = False
        for x in self.actors:
            x.cpu()
            x.low_action_bound.cpu()
            x.high_action_bound.cpu()
        for x in self.critics:
            x.cpu()
        for x in self.actors_target:
            x.cpu()
            x.low_action_bound.cpu()
            x.high_action_bound.cpu()
        for x in self.critics_target:
            x.cpu()

    def update_policy(self, prioritized=False):
        # do not train until exploration is enough
        # print 'update'
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            print time.time(), "A", agent
            transitions = self.memory.sample(
                self.batch_size, prioritized=prioritized)
            batch = self.memory.Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = Variable(th.stack(batch.states).type(FloatTensor))
            action_batch = Variable(th.stack(batch.actions).type(FloatTensor))
            reward_batch = Variable(th.stack(batch.rewards).type(FloatTensor))
            print time.time(), "B", agent
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = Variable(th.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor))

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state, whole_action)

            non_final_next_actions = [
                self.actors_target[i](non_final_next_states[:,
                                                            i,
                                                            :]) for i in range(
                                                                self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
#            non_final_next_actions = Variable(non_final_next_actions)
            non_final_next_actions = (
                non_final_next_actions.transpose(0,
                                                 1).contiguous())

            target_Q = Variable(th.zeros(
                self.batch_size).type(FloatTensor))
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1,
                                            self.n_agents * self.n_actions))

            # scale_reward: to scale reward in Q functions
            target_Q = (target_Q * self.GAMMA) + (
                reward_batch[:, agent] * scale_reward).view(-1)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            action_i.retain_grad()
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()

            params = action_i[:, 4:]
            high = self.actors[agent].high_action_bound
            high = high.repeat(action_i.size()[0], 1)
            low = self.actors[agent].low_action_bound
            low = low.repeat(action_i.size()[0], 1)

            if params.data.type() == 'torch.cuda.FloatTensor':
                high = high.cuda()
                low = low.cuda()

            pmax = ((high - params) / (high - low))
            pmin = ((params - low) / (high - low))

            grad = action_i.grad[:, 4:]
            g1 = (grad < 0).float() * pmin
            g2 = (grad >= 0).float() * pmax

            action_i.grad = th.cat([action_i.grad[:, :4], (g1 + g2)], 1)

            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
        # print c_loss, a_loss
        return c_loss, a_loss

    def select_action(self, state_batch, i):
        act = self.actors[i](state_batch.view(1, -1))
        # self.steps_done += 1
        return act[0], [None, None]

    def critic_predict(self, state_batch, action_batch, i):
        return self.critics[i](state_batch, action_batch)
