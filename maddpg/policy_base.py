import torch.nn as nn
import torch
from typing import List
import numpy as np

class BasePolicy:
    def __init__(self, device, adv_type):
        self.policy = None
        self.device = device
        self.storage = []
        self.batched_storage = []
        self.adv_type = adv_type

    def parameters(self):
        return self.policy.parameters()

    def named_parameters(self):
        return self.policy.named_parameters()

    def forward(self, observation, step, neighbors):
        return self.policy.forward(observation, step, neighbors)

    def add_to_memory(self, experience):
        self.storage.append(experience)

    def clear_memory(self):
        del self.storage[:]
        del self.batched_storage[:]

    def state_dict(self):
        return self.policy.state_dict()

    def set_batched_storage(self, BATCH_SIZE):
        if len(self.storage) == 0:
            raise Exception("Nothing in storage! Cant perform backprop!")

        self.batched_storage = []
        for actual_batch_ix in range(0, BATCH_SIZE):
            experience_list = []
            seen_none = False
            for step in range(0, len(self.storage)):
                batch_experience = self.storage[step][actual_batch_ix]
                if batch_experience is not None:
                    if seen_none:
                        raise Exception('Something wrong with storage %s' % (self.storage))
                    experience_list.append(batch_experience)
                else:
                    seen_none = True
            self.batched_storage.append(experience_list)

    def compute_loss(self, gamma = 0.99, eps=1e-5, standardize_rewards=False):
        if len(self.storage) == 0:
            raise Exception("Nothing in storage! Can't perform backprop!")
        BATCH_SIZE = len(self.batched_storage)
        if BATCH_SIZE == 0:
            raise Exception('Error here, no batches')

        mean_reward = np.zeros((BATCH_SIZE))
        length_iteration = np.zeros((BATCH_SIZE))
        loss = []
        for batch_ix in range(0, BATCH_SIZE):
            experience_list = self.batched_storage[batch_ix]
            length_of_iteration_for_batch = len(experience_list)
            R = 0
            returns = []
            l1_loss_func = nn.SmoothL1Loss()
            # 对经验倒序循环处理
            for r in experience_list[::-1]:
                R = r.rewards + gamma * R
                returns.append([R])
            returns.reverse()
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            mean_rewards_for_batch = returns.mean().item()
            if standardize_rewards:
                returns = (returns - returns.mean())/(returns.std() + 1e-5)
            log_probs = torch.vstack([experience.log_prob for experience in experience_list]).to(self.device)
            state_vals = torch.vstack([experience.state_val for experience in experience_list]).to(self.device)
            if self.adv_type == 'normal':
                adv = (returns - state_vals)
            elif self.adv_type == 'clamped':
                adv = (returns - state_vals.detach()).clamp(min=0)
            elif self.adv_type == 'clamped_q':
                adv = (returns).clamp(min=0)
            else:
                raise Exception(self.adv_type + " isnt supported")
            actor_loss = torch.mean(-log_probs*(adv), dim = 0)
            critic_loss = torch.mean(l1_loss_func(state_vals, returns), dim = 0)
            loss_for_batch = actor_loss + critic_loss
            mean_reward[batch_ix] = mean_rewards_for_batch
            length_iteration[batch_ix] = length_of_iteration_for_batch
            loss.append(loss_for_batch)
        mean_loss_across_batches = torch.cat(loss).mean()
        return mean_loss_across_batches, mean_reward, length_iteration

    def consensus_update(self, neighbors_vnet: List[nn.Module]):
        current_vnet_dict = self.policy.v_net.state_dict()
        for neighbor in neighbors_vnet:
            for k in current_vnet_dict.keys():
                current_vnet_dict[k] += neighbor[k]
        for param_name in current_vnet_dict.keys():
            current_vnet_dict[param_name] /= len(neighbors_vnet)
        self.policy.v_net.load_state_dict(current_vnet_dict)