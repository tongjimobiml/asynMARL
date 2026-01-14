import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import random

from utils.util import update_linear_schedule
from runner.base_runner import Runner
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class MSDRunner(Runner):
    def __init__(self, config):
        super(MSDRunner, self).__init__(config)
       
    def run(self):
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_time = 0.0
        num_queries = 0
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)
            
            self.envs.reset()
            self.agent_buffer_status = {agent_id: False for agent_id in range(self.num_agents)}
            for step in range(self.episode_length):
                # Observe: get state, obs and queries
                queries = self.envs.get_queries()

                if len(queries) > 0:
                    # print("Number of queries: {}".format(len(queries)))

                    # Sample actions
                    values, actions, action_log_probs, actions_env, all_query_input, step_time = self.collect(queries)
                    num_queries += len(queries)
                    total_time += step_time
                    # Obser reward and next obs
                    # new_state, new_obs, rewards, dones, infos, veh_features = self.envs.step(actions)
                    experiences = self.envs.step(actions, values, action_log_probs, all_query_input)
                    # state, new_state, obs, new_obs, rewards, dones, infos = self.envs.step(actions)

                    # next_values = self.get_next_values(new_state, new_obs, queries, veh_features)

                    # data = queries, state, obs, rewards, dones, infos, values, next_values, actions, action_log_probs, rnn_states, rnn_states_critic
                    # data = state, new_state, obs, new_obs, rewards, dones, infos, values, next_values, actions, action_log_probs, rnn_states, rnn_states_critic


                    # insert data into buffer
                    if episode < self.max_train_episode:
                        self.insert(experiences)
                else:
                    self.envs.step(None)

            if self.all_args.network == 'koln' and episode % 10 == 9:
                self.envs.close()
            self.envs.summary()
            # compute return and update network

            for agent_id in range(self.num_agents):
                self.buffer[agent_id].before_update()

            self.compute()

            if episode < self.max_train_episode:
                train_infos = self.train()
                # post process
                total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

                # save model
                if (episode % self.save_interval == 0 or episode == episodes - 1):
                    self.save()

                # log information
                if episode % self.log_interval == 0:
                    end = time.time()
                    print("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                          .format(self.algorithm_name,
                                  self.experiment_name,
                                  episode,
                                  episodes,
                                  total_num_steps,
                                  self.num_env_steps,
                                  int(total_num_steps / (end - start))))

                    self.log_train(train_infos, total_num_steps)

            else:
                train_infos = []
                for agent_id in range(self.num_agents):
                    train_info = dict()
                    train_info['value_loss'] = 0
                    train_info['policy_loss'] = 0
                    train_info['dist_entropy'] = 0
                    train_info['actor_grad_norm'] = 0
                    train_info['critic_grad_norm'] = 0
                    train_info['ratio'] = 0
                    train_infos.append(train_info)

            # eval
            # if episode % self.eval_interval == 0 and self.use_eval:
            #     self.eval(total_num_steps)
        # self.envs.close()
        return

    def warmup(self):
        # reset env
        self.envs.reset()

    @torch.no_grad()
    def collect(self, queries):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        all_query_input = []

        agent_query_list = []
        for agent_id in range(self.num_agents):
            agent_query_list.append([])
        for query in queries:
            agent_id = query['agent']
            agent_query_list[agent_id].append(query['observation'][-1].copy())

        step_time = 0.0

        agent_queries = {}
        for agent_id in range(self.num_agents):
            if len(agent_query_list[agent_id]) > self.max_query_cnt:
                idx_list = random.sample(range(0, len(agent_query_list[agent_id])), self.max_query_cnt)
                # np.array: [seq_len, query_feature_dim]
                agent_queries[agent_id] = np.array([agent_query_list[agent_id][i] for i in idx_list])
            else:
                agent_queries[agent_id] = np.array(agent_query_list[agent_id].copy())

        # for agent_id in range(self.num_agents):
        for query in queries:
            agent_id = query['agent']
            step = 0  # delete later
            self.trainer[agent_id].prep_rollout()
            # if not self.agent_buffer_status[agent_id]:
            #     agent_obs = np.array(list(obs[agent_id]) + query['source_feature'] + query['dest_feature'])
            #     agent_obs = np.expand_dims(agent_obs, axis=0)
            #     self.buffer[agent_id].obs.append(agent_obs.copy())
            #
            #     self.buffer[agent_id].share_obs.append(np.expand_dims(state.copy(), axis=0))
            #     self.agent_buffer_status[agent_id] = True


            road_obs = np.array(query['observation'][:-1], dtype=np.float32).flatten()
            query_obs = np.array(query['observation'][-1], dtype=np.float32)
            obs = np.concatenate([road_obs, query_obs])
            all_query_input.append(agent_queries[agent_id])
            query_features = [agent_queries[agent_id]]

            if self.use_centralized_V:
                # share_obs = state
                # share_obs = np.concatenate([query['state'], query['observation']])
                share_obs = np.concatenate([query['state'], obs])
            else:
                # share_obs = np.array(obs[agent_id] + query['source_feature'] + query['dest_feature'])
                # share_obs = np.array(query['observation'])
                share_obs = obs
            t1 = time.time()
            value, action, action_log_prob \
                = self.trainer[agent_id].policy.get_actions(share_obs,
                                                            obs,
                                                            query_features,
                                                            self.buffer[agent_id].masks[step],
                                                            query['available_actions'])
            t2 = time.time()
            step_time += t2 - t1
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)

            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                # action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 0)
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))


        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0)
        values = np.expand_dims(values, axis=2)
        actions = np.array(actions).transpose(1, 0)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)

        return values, actions, action_log_probs, actions_env, all_query_input, step_time/len(queries)

    def get_next_value(self, agent_id, state, obs, mask):
        if self.use_centralized_V:
            # critic_input = np.array(state)
            critic_input = np.concatenate([state, np.array(obs[:-1]).flatten(), np.array(obs[-1])])
        else:
            # critic_input = np.array(obs)
            critic_input = np.concatenate([np.array(obs[:-1]).flatten(),
                                           np.array(obs[-1])])
        next_value = self.trainer[agent_id].policy.get_values(critic_input,
                                                              mask)
        next_value = _t2n(next_value)
        return next_value

    def insert(self, experiences):
        # queries, state, obs, rewards, dones, infos, values, next_values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        # masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        dones = np.array([e['done'] for e in experiences])
        masks = np.ones((len(experiences), 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        masks = masks.reshape(self.n_rollout_threads, len(experiences), 1)
        # share_obs = []
        # for o in obs:
        #     share_obs.append(list(chain(*o)))
        # share_obs = np.array(share_obs)

        for i, experience in enumerate(experiences):
            agent_id = experience['agent']
            # for agent_id in range(self.num_agents):
            #     if not self.use_centralized_V:
            #         share_obs = np.array(list(obs[agent_id]))
            # agent_obs = np.array(list(obs[agent_id]) + query['source_feature'] + query['dest_feature'])
            # agent_obs = np.array(experience['last_obs'])
            last_obs = np.concatenate([np.array(experience['last_obs'][:-1]).flatten(),
                                        np.array(experience['last_obs'][-1])])
            last_obs = np.expand_dims(last_obs, axis=0)
            query_obs = experience['last_query_obs']
            if self.use_centralized_V:
                # share_obs = np.expand_dims(experience['state'], axis=0)
                # share_obs = np.concatenate([experience['last_state'], np.array(experience['last_obs'])], axis=0)
                share_obs = np.concatenate([experience['last_state'], 
                                            np.array(experience['last_obs'][:-1]).flatten(), 
                                            np.array(experience['last_obs'][-1])], axis=0)
                # 去掉了一些冗余的路段特征
                share_obs = np.expand_dims(share_obs, axis=0)
            else:
                share_obs = last_obs.copy()

            self.buffer[agent_id].insert(share_obs=share_obs,
                                         obs=last_obs,
                                         query_obs=query_obs,
                                         actions=experience['last_action'],
                                         action_log_probs=experience['last_action_prob'],
                                         value_preds=experience['last_value'],
                                         next_value_preds=self.get_next_value(agent_id,
                                                                              experience['state'],
                                                                              experience['obs'],
                                                                              masks[:, i]),
                                         rewards=np.array(experience['reward']),
                                         masks=masks[:, i],
                                         available_actions=experience['available_actions'])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  
