from model import Critic, Actor, MetaActor
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience, ExperienceOptions
from torch.optim import Adam
from randomProcess import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from params import scale_reward
import time
import os
from collections import namedtuple
import time
from scipy.cluster.vq import kmeans


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
                       for i in range(n_options)]
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act)]
        self.meta_actor = MetaActor(n_agents, dim_obs, n_options)

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        self.meta_actor_target = deepcopy(self.meta_actor)

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
        self.meta_optimizer = Adam(self.meta_actor.parameters(), lr=0.0001)
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        if self.use_cuda:
            self.meta_actor.cuda()
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
        self.meta_actor.cuda()
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
        self.meta_actor.cpu()
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
            print("UPDATETIME", time.time() - t0)
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []
        transitions = self.memory.sample(
            self.batch_size, prioritized=prioritized)
        batch = ExperienceOptions(*zip(*transitions))
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
        whole_option = option_batch.view(self.batch_size, -1)
        whole_reward = reward_batch.view(self.batch_size, -1)

        self.meta_optimizer.zero_grad()
        self.critic_optimizer[0].zero_grad()
        for opt in range(self.n_options):
            self.actor_optimizer[opt].zero_grad()

        # loss_Q.backward()
        loss_Q = 0
        actor_loss_total = 0
        # self.critic_optimizer[0].step()

        for agent in range(self.n_agents):
            meta_state = state_batch[:, agent]
            meta_state_1 = non_final_next_states[:, agent]
            meta_option = option_batch[:, agent]

            index = th.LongTensor([agent, 1 - agent])
            state_batch_ordered = state_batch.clone()
            action_batch_ordered = action_batch.clone()
            state_batch_ordered[:, index] = state_batch_ordered
            action_batch_ordered[:, index] = action_batch_ordered
            whole_state = state_batch_ordered.view(self.batch_size, -1)
            whole_action = action_batch_ordered.view(self.batch_size, -1)

            current_Q = self.critics[0](whole_state, whole_action)

            non_final_next_actions = [self.select_action(non_final_next_states[:,
                                                                               i,
                                                                               :])[0] for i in range(self.n_agents)]

            non_final_next_actions = th.stack(non_final_next_actions)
    #            non_final_next_actions = Variable(non_final_next_actions)
            non_final_next_actions = (
                non_final_next_actions.transpose(0,
                                                 1).contiguous())

            non_final_next_states_ordered = non_final_next_states.clone()
            non_final_next_actions_ordered = non_final_next_actions.clone()

            non_final_next_states_ordered[
                :, index] = non_final_next_states_ordered
            non_final_next_actions_ordered[
                :, index] = non_final_next_actions_ordered

            target_Q = Variable(th.zeros(
                self.batch_size).type(FloatTensor))
            target_Q[non_final_mask] = self.critics_target[0](
                non_final_next_states_ordered.view(-1,
                                                   self.n_agents * self.n_states),
                non_final_next_actions_ordered.view(-1,
                                                    self.n_agents * self.n_actions))

            # scale_reward: to scale reward in Q functions
            target_Q = (target_Q * self.GAMMA) + (
                reward_batch[:, agent] * scale_reward).view(-1)

            loss_Q += nn.MSELoss()(current_Q, target_Q.detach())

            pred_options, encodings = self.meta_actor(meta_state)
            meta_entropy = (-1 * th.sum((pred_options *
                                         pred_options.log()), dim=1)).mean()

            ########################################################
            # CLUSTER LOSS
            ########################################################
            # clusters, _ = kmeans(encodings.data.numpy(), 2)
            # clusters = Variable(th.from_numpy(clusters).float())
            # diff1 = (encodings - clusters[0]).mean(dim=1)
            # diff2 = (encodings - clusters[1]).mean(dim=1)
            # diff = (th.stack([diff1, diff2], dim=1)).abs()
            # _, diff = th.min(diff, dim=1)
            # _, pred_option_max = th.max(pred_options, dim=1)
            # match_clusters = (diff == pred_option_max).float()
            # meta_loss = match_clusters.mean(dim=0) + (-0.1 * entropy)
            meta_loss = (-0.1 * meta_entropy)
            ########################################################

            # loss_Q = nn.MSELoss()(current_Q, nextq.detach())
            # loss_Q.backward()
            # meta_loss.backward(retain_graph=True)
            # self.meta_optimizer.step()

            opt_acts = []
            for opt in range(self.n_options):
                action_i = self.actors[opt](meta_state)
                opt_acts.append(action_i)
            
            action = th.stack(opt_acts, dim=1)
            w = pred_options.unsqueeze(1)
            act = w.bmm(action)
            act = act.squeeze(1)

            act.retain_grad()
            ac = action_batch_ordered.clone()
            ac[:, 0, :] = act
            whole_action_new = ac.view(self.batch_size, -1)
            actor_loss = - \
                self.critics[0](state_batch_ordered.view(
                    self.batch_size, -1), whole_action_new)

            actor_loss_total += actor_loss.mean() + meta_loss

        loss_Q.backward()
        self.critic_optimizer[0].step()

        actor_loss_total.backward()

        params = act[:, 4:]
        high = self.actors[opt].high_action_bound
        high = high.repeat(act.size()[0], 1)
        low = self.actors[opt].low_action_bound
        low = low.repeat(act.size()[0], 1)

        if params.data.type() == 'torch.cuda.FloatTensor':
            high = high.cuda()
            low = low.cuda()

        pmax = ((high - params) / (high - low))
        pmin = ((params - low) / (high - low))

        grad = act.grad[:, 4:]
        g1 = (grad < 0).float() * pmin
        g2 = (grad >= 0).float() * pmax

        act.grad = th.cat([act.grad[:, :4], (g1 + g2)], 1)

        for opt in range(self.n_options):
            self.actor_optimizer[opt].step()
        self.meta_optimizer.step()

        print("UPDATETIME POLICIES", time.time() - t0)
        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(len(self.critics)):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
            for i in range(len(self.actors)):
                soft_update(self.actors_target[i], self.actors[i], self.tau)
            soft_update(self.meta_actor_target, self.meta_actor, self.tau)

        print("UPDATETIME", time.time() - t0)
        # print c_loss, a_loss
        return loss_Q, actor_loss_total

    def select_action(self, state_batch, i=None):
        act1 = self.actors_target[0](state_batch)
        act2 = self.actors_target[1](state_batch)
        act_raw = th.stack([act1, act2], dim=1)
        w, _ = self.meta_actor_target(state_batch)
        w2 = w.unsqueeze(1)
        act = w2.bmm(act_raw)
        act = act.squeeze(1)

        _, opt = w.max(1)
        # self.steps_done += 1
        return act, int(opt[0].data.numpy())

    def critic_predict(self, state_batch, action_batch, i):
        return self.critics[0](state_batch, action_batch)


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
            transitions = self.memory.sample(
                self.batch_size, prioritized=prioritized)
            batch = Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = Variable(th.stack(batch.states).type(FloatTensor))
            action_batch = Variable(th.stack(batch.actions).type(FloatTensor))
            reward_batch = Variable(th.stack(batch.rewards).type(FloatTensor))
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
        return act[0]

    def critic_predict(self, state_batch, action_batch, i):
        return self.critics[i](state_batch, action_batch)
