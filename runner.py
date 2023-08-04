from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from maddpg.actor_critic import Fitting
import pickle

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        # self.fitting_networks = self._init_fittings()

        # model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        # for agent in self.agents:
        #     file_path = os.path.join(model_path, 'agent_%d' % agent.agent_id, num + '_actor_params.pkl')
        #     parameters = torch.load(file_path)
        #     agent.policy.actor_network.load_state_dict(parameters)
        #     file_path = os.path.join(model_path, 'agent_%d' % agent.agent_id, num + '_critic_params.pkl')
        #     parameters = torch.load(file_path)
        #     agent.policy.critic_network.load_state_dict(parameters)

        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.device = args.device

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    # def _init_fittings(self):
    #     fitting_networks = {}
    #     for id in range(self.args.n_agents):
    #         fitting_networks[id] = Fitting().to(self.args.device)
    #         fitting_networks[id].copy_weights(self.agents[id].policy.actor_network)
    #     return fitting_networks
    #
    # def fitting_update(self):
    #     for id in range(self.args.n_agents):
    #         self.fitting_networks[id].copy_weights(self.agents[id].policy.actor_target_network)
    #     return

    def get_init_actions(self, observations):
        policy_initial = []
        for id, agent in enumerate(self.agents):
            policy_initial.append(agent.policy.actor_network(observations[id]))
        return policy_initial

    def get_actions(self, observations, agents, noise_rate = 0, epsilon = 0):
        policy_initial = []
        policies = []
        actions = []
        actions_initial = []
        for agent in agents:
            policies.append(agent.policy.actor_network)
        for i,policy in enumerate(policies):
            inputs = torch.tensor(observations[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            initial_policy_distribution = policy.forward(inputs, 0, None)
            policy_initial.append(initial_policy_distribution)
            actions_initial.append(initial_policy_distribution.to('cpu'))

        for i,policy in enumerate(policies):
            if np.random.uniform() < epsilon:
                latent_vector = np.random.uniform(-self.args.high_action, self.args.high_action,
                                                                self.args.action_shape[0])
                latent_vector = torch.from_numpy(latent_vector).float()
            else:
                latent_vector = policy.forward(policy_initial[i], 1, policy_initial)
                latent_vector = policy.forward(latent_vector, 2).squeeze(0).to('cpu')
            actions.append(latent_vector.cpu().numpy())

        for action in actions:
            noise = noise_rate * self.args.high_action * np.random.randn(*action.shape)  # gaussian noise
            action += noise
            action = np.clip(action, -self.args.high_action, self.args.high_action)
        return actions, actions_initial

    def run(self):
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            with torch.no_grad():
                action, actions_initial = self.get_actions(s, self.agents, self.noise, self.epsilon)
                u = action
                actions = action

            s_next, r, done, info = self.env.step(actions)
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents], actions_initial)
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                fitting_list = []
                # self.fitting_update()
                for agent in self.agents:
                    fitting_list.append(agent.policy.actor_target_network)

                for agent in self.agents:
                    # policies = [a.policy for a in self.agents]
                    agent.learn(transitions, fitting_list)

            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
                plt.cla()
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
            np.save(self.save_path + '/returns.pkl', returns)

    def evaluate(self):
        returns = []
        if self.args.evaluate:
            model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
            num = '120'
            for agent in self.agents:
                file_path = os.path.join(model_path, 'agent_%d' % agent.agent_id, num + '_actor_params.pkl')
                parameters = torch.load(file_path)
                agent.policy.actor_network.load_state_dict(parameters)

        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            if self.args.evaluate:
                reward_list = []


            for time_step in range(self.args.evaluate_episode_len):
                # self.env.render()
                # actions = []
                with torch.no_grad():
                   actions, latents = self.get_actions(s, self.agents, 0, 0)

                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)

                if self.args.evaluate:
                    reward_list.append(r)
                rewards += r[0]
                s = s_next
            if self.args.evaluate:
                file_path = os.path.join(model_path,'episode%d'%episode+'_rewards.txt')
                with open(file_path, 'w') as f:
                    for r in reward_list:
                        f.write("%s\n"%r)
                f.close()

            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
