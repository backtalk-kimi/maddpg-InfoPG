import torch
import os
from actor_critic import Actor, Critic, Fitting
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from scipy import stats
# from torch.distributions import Categorical

class MADDPG():
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分

        # super(MADDPG, self).__init__(args)

        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        device = args.device
        self.device = device

        #critic_type select
        self.critic_type = 'MA_short'
        # critic_type = 'MA_long'
        # critic_type = 'MADDPG'

        # create the network
        self.actor_network = Actor(device, args)
        self.critic_network = Critic(args, self.critic_type).to(device)

        # build up the target network
        self.actor_target_network = Actor(device, args)
        self.critic_target_network = Critic(args, self.critic_type).to(device)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)



        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

       # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def select_action(self, observations, actor, fittings_network):
        policy_initial = []
        for id in range(self.args.n_agents):
            inputs = observations[id]
            if id == self.agent_id:
                initial_policy_distribution = actor.forward(inputs,0,None)
            else:
                initial_policy_distribution = fittings_network[id].forward(inputs, 0)
            policy_initial.append(initial_policy_distribution)

        # action_initial = policy_initial
        policy_com = actor.forward(policy_initial[self.agent_id], 1, policy_initial)
        policy_final = actor.forward(policy_com, 2, None)
        return policy_final, policy_initial

    # update the network
    def train(self, transitions, fitting_networks):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32).to(self.device)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        u_initial, target_policies = [], []
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id].detach())
            u.append(transitions['u_%d' % agent_id].detach())
            o_next.append(transitions['o_next_%d' % agent_id].detach())
            u_initial.append(transitions['u_initial_%d' % agent_id].detach())
            # if agent_id != self.agent_id:
            #     self.fitting_network[agent_id].copy_weights(policies[agent_id].actor_target_network)
            #     target_policies.append(self.fitting_network[agent_id])
            # else:
            #     target_policies.append(self.actor_target_network)
        # critic_loss
        actions_next, latent_next = self.select_action(o_next, self.actor_target_network, fitting_networks)
        q_value = self.critic_network(o, u_initial, u[self.agent_id])
        q_next = self.critic_target_network(o_next, latent_next, actions_next).detach()
        target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()
        critic_loss = (target_q - q_value).pow(2).mean()
        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        actions, latent = self.select_action(o, self.actor_network, fitting_networks)
        actor_loss = - self.critic_network(o, latent, actions).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')

    def fitting_plt_draw(self, id):
        plt.plot(range(len(self.fitting_evaluation[id])), self.fitting_evaluation[id])
        plt.xlabel('fitting' + str(id))
        plt.ylabel('evaluation')
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        plt.savefig(model_path + '/' + str(id) + '_plt.png', format='png')
        plt.cla()

    def save_fitting_model(self, id):
        num = str(self.train_step // 10000)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.fitting_network[id].state_dict(), model_path + '/' + num + '_' + str(id) + '_fitting_params.pkl')
        # torch.save(self.actor_network.state_dict(), model_path + '/' + 'final_actor_params.pkl')

