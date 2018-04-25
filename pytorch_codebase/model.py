import torch as th
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent

        self.FC1 = nn.Linear(obs_dim, 1024)
        self.FC2 = nn.Linear(1024 + act_dim, 512)
        self.FC3 = nn.Linear(512, 256)
        self.FC4 = nn.Linear(256, 128)
        self.FC5 = nn.Linear(128, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = th.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        result = F.relu(self.FC3(result))
        return self.FC5(F.relu(self.FC4(result)))


class MetaCritic(nn.Module):

    def __init__(self, n_agent, dim_observation, n_option):
        super(MetaCritic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        obs_dim = dim_observation
        act_dim = n_option

        self.FC1 = nn.Linear(obs_dim, 1024)
        self.FC2 = nn.Linear(1024 + act_dim, 512)
        self.FC3 = nn.Linear(512, 256)
        self.FC4 = nn.Linear(256, 128)
        self.FC5 = nn.Linear(128, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, opts):
        result_state = F.relu(self.FC1(obs))
        combined = th.cat([result_state, opts], 1)
        resulta = F.relu(self.FC2(combined))
        result = F.relu(self.FC3(resulta))
        result = F.relu(self.FC4(result))
        return self.FC5(result), result


class Actor(nn.Module):

    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.low_action_bound = Variable(th.FloatTensor(
            np.array([0., -180., -180., -180., 0., -180.])).view(1, -1))
        self.high_action_bound = Variable(th.FloatTensor(
            np.array([100., 180., 180., 180., 100., 180.])).view(1, -1))
        self.leakyrelu = nn.LeakyReLU()

        self.FC1 = nn.Linear(dim_observation, 1024)
        self.FC2 = nn.Linear(1024, 512)
        self.FC3 = nn.Linear(512, 256)
        self.FC4 = nn.Linear(256, 128)
        self.FC5 = nn.Linear(128, dim_action)

    # action output between -2 and 2
    def forward(self, obs):
        result = self.leakyrelu(self.FC1(obs))
        result = self.leakyrelu(self.FC2(result))
        result = self.leakyrelu(self.FC3(result))
        result = self.leakyrelu(self.FC4(result))
        result = self.FC5(result)
        r1 = result[:, :4]
        r2 = result[:, 4:]

        r1 = F.softmax(r1)
        r2 = F.sigmoid(r2)

        if r2.data.type() == 'torch.cuda.FloatTensor':
            r2 = (r2 * (self.high_action_bound.cuda() -
                        self.low_action_bound.cuda())) + self.low_action_bound.cuda()
        else:
            r2 = (r2 * (self.high_action_bound - self.low_action_bound)
                  ) + self.low_action_bound
        out = th.cat((r1, r2), 1)

        return out
