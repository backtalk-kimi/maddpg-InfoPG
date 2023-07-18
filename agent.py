import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)
        self.agent_num = 3

    def select_action(self, observations, agents):
        policy_initial = []
        policies = []
        k_levels = 1
        for agent in range(self.agents):
            policies.append(agent.policy.actor_network)
        for agent in range(self.AGENT_NAMES):
            initial_policy_distribution, state_val = agents[agent].policy.forward(observations[agent], 0, None)
            policy_initial.append(initial_policy_distribution)

        actions = []

        for k in range(0, k_levels):
            output_dist = []
            for agent in range(self.agents):
                neighbors_policy = []
                for neighbor in range(self.agents):
                    if agent.agent_id != neighbor.agent_id:
                        neighbors_policy.append([policy_initial[neighbor.agent_id], neighbor.agent_id])
                latent_vector = agent.forward(policy_initial[agent], 1, neighbors_policy)
                output_dist.append = latent_vector
            policy_initial = output_dist

        for i, agent in enumerate(self.agents):
            final_policy_distribution = agent.forward(policy_initial[i], 2, None).to('cpu')
            distribution = Categorical(probs=final_policy_distribution)
            batch_action = distribution.sample()
            actions.append(batch_action.item())
        return actions

    def learn(self, transitions, fitting_networks):
        self.policy.train(transitions, fitting_networks)

