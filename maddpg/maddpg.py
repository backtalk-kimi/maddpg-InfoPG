import torch
import os
from maddpg.actor_critic import Actor, Critic
import matplotlib.pyplot as plt


class MADDPG:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor(args, agent_id)
        self.critic_network = Critic(args)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id)
        self.critic_target_network = Critic(args)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())


        agents_id = [i for i in range(args.n_agents)]
        agents_id.remove(agent_id)
        self.fitting_network = {}
        self.fitting_optim = {}
        self.fitting_evaluation = {}
        for id in agents_id:
            self.fitting_network[id] = Actor(args, id)
            self.fitting_optim[id] = torch.optim.Adam(self.fitting_network[id].parameters(), lr=self.args.lr_actor)
            self.fitting_evaluation[id] = []

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

        # 加载模型
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # update the network
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        # calculate the target Q value function
        u_next = []

            # 得到下一个状态对应的动作
        for agent_id in range(self.args.n_agents):
            if agent_id == self.agent_id:
                with torch.no_grad():
                    u_next.append(self.actor_target_network(o_next[agent_id]))
            else:
                u_next.append(self.fitting_network[agent_id](o_next[agent_id]))
        q_next = self.critic_target_network(o_next, u_next).detach()

        target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # # calculate the target Q value function
        # u_next = []
        # with torch.no_grad():
        #     # 得到下一个状态对应的动作
        #     index = 0
        #     for agent_id in range(self.args.n_agents):
        #         if agent_id == self.agent_id:
        #             u_next.append(self.actor_target_network(o_next[agent_id]))
        #         else:
        #             # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
        #             u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
        #             index += 1
        #     q_next = self.critic_target_network(o_next, u_next).detach()
        #
        #     target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = - self.critic_network(o, u).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # the fitting loss
        for id in self.fitting_network:
            with torch.no_grad():
                for agent in other_agents:
                    if agent.agent_id == id:
                        u_i = agent.policy.actor_target_network(o_next[id])
                        break
            fitting_loss = (u_i - u_next[id]).pow(2).mean()
            self.fitting_optim[id].zero_grad()
            fitting_loss.backward()
            self.fitting_optim[id].step()

            if self.train_step > 0 and self.train_step % 1000 == 0:
                fitting_loss_value = fitting_loss.tolist()
                self.fitting_evaluation[id].append(fitting_loss_value)


        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

        if self.train_step > 0 and self.train_step == 20000:
            self.fitting_plt_draw()

        if self.train_step >= 1000000 and self.train_step % 10000 == 0:
            self.save_fitting_model()

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

    def fitting_plt_draw(self):
        for id in self.fitting_network:
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

    def save_fitting_model(self):
        num = str(self.train_step // 10000)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for id in self.fitting_network:
            torch.save(self.fitting_network[id].state_dict(), model_path + '/' + num + '_' + str(id) + '_fitting_params.pkl')
        torch.save(self.actor_network.state_dict(), model_path + '/' + 'final_actor_params.pkl')

