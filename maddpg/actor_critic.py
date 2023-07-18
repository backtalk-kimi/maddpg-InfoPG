import torch
import torch.nn.functional as F
import numpy as np
from typing import List
import torch.nn as nn

class Fitting(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sampler = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Linear(64, 5),
            # nn.Linear(25, 5),
            nn.Tanh(),
        )

    def copy_weights(self, source_model):
        source_state_dict = source_model.state_dict()
        down_sampler_dict = self.down_sampler.state_dict()
        policy_dict = self.policy.state_dict()

        for name, param in source_state_dict.items():
            if name.startswith('policy.down_sampler'):
                param_name = name.replace('policy.down_sampler.', '')
                down_sampler_dict[param_name].copy_(param)
            elif name.startswith('policy.policy'):
                param_name = name.replace('policy.policy.', '')
                policy_dict[param_name].copy_(param)

    def forward(self, observation, step = 0):
        encoded = self.down_sampler(observation)
        policy_distribution = self.policy(encoded)
        return policy_distribution


class Actor(nn.Module):
    def __init__(self, device, args, constant=False):
        super(Actor, self).__init__()
        self.policy = Piston_PolicyHelperCASE(args)
        if constant:
            for name, param in self.policy.named_parameters():
                nn.init.constant_(param, 0.5)
        # if model_state_dict is not None:
        #     self.policy.load_state_dict(model_state_dict)
        self.policy.to(device)
    def forward(self, observation, step, neighbors = None):
        return self.policy.forward(observation, step, neighbors)

class Piston_PolicyHelperCASE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_action = args.high_action
        self.k_level = 1
        self.down_sampler = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Linear(64, 5),
            nn.Tanh(),
        )
        self.to_probs = nn.Sequential(
            nn.Linear(5, 5),
            # nn.ReLU(),
            nn.Tanh()
        )
        self.recurr_policy = nn.GRUCell(input_size= 5, hidden_size= 5)


    def forward(self, observation, step, neighbors=None):
        if step == 0:
            return self.forward_initial(observation)
        elif step == 1:
            return self.forward_communicate(observation, neighbors)
        elif step == 2:
            return self.to_probs(observation)
        else:
            raise Exception('Incorrect step number for forward prop, should be: 0,1,2')

    def forward_initial(self, observation):
        encoded = self.down_sampler(observation)
        policy_distribution = self.max_action * self.policy(encoded)
        return policy_distribution

    def forward_communicate(self, policy_dist, neighbors):
        batch_outputs = []
        for time in range(self.k_level - 1):
            communicate = []
            for i, neighbor1 in enumerate(neighbors):
                hx = neighbor1.clone()
                for j, neighbor2 in enumerate(neighbors):
                    if i != j:
                        hx = self.recurr_policy(neighbor2, hx)
                communicate.append(hx)
            neighbors = communicate

        hx = policy_dist
        for neighbor in neighbors:
            neighbor_dists = neighbor.clone().detach()
            assert hx.shape == neighbor_dists.shape, '%s and %s' % (hx.shape, neighbor_dists.shape)
            hx = self.recurr_policy(neighbor_dists, hx)
        batch_outputs.append(hx)
        polciy_com = torch.cat(batch_outputs, dim=0)
        return polciy_com

    def forward_probs(self, latent_vector):
        probs = self.to_probs(latent_vector)
        return probs


class Critic(nn.Module):
    def __init__(self, args, critic_type):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        if critic_type == 'MA_long':
            self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape) + 5, 64)
        elif critic_type == 'MA_short':
            self.fc1 = nn.Linear(sum(args.obs_shape) + 5, 64)
        elif critic_type == 'MADDPG':
            self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)
        self.critic_type = critic_type
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action = 0, latent = 0):
        state = torch.cat(state, dim=1)

        if self.critic_type == 'MA_long':
            for i in range(len(action)):
                action[i] /= self.max_action
            latent = latent / self.max_action
            action = torch.cat(action, dim=1)
            x = torch.cat([state, action, latent], dim=1)

        elif self.critic_type == 'MA_short':
            latent = latent/ self.max_action
            x = torch.cat([state, latent], dim=1)

        elif self.critic_type == 'MADDPG':
            for i in range(len(action)):
                action[i] /= self.max_action
            action = torch.cat(action, dim=1)
            x = torch.cat([state, action], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value

