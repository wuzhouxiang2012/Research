import numpy as np
import torch
import os
from env_for_evaluate import Environment
from distribution import Distribution
from request import Request

class Evaluation():
    def __init__(self, num_episode=10):
        self.num_episode = num_episode
             # create environment
        self.dist1 = Distribution(id=0, vals=[2], probs=[1])
        self.dist2 = Distribution(id=1, vals=[5], probs=[1])
        self.dist3 = Distribution(id=2, vals=[2,8], probs=[0.5,0.5])

        self.env_list = self.produce_env_list()
        self.always_reject_avg_reward, self.always_reject_avg_acc_rate = self.always_reject()
        self.always_accept_avg_reward, self.always_accept_avg_acc_rate = self.always_accept()
        self.reject_when_full_avg_reward, self.reject_when_full_avg_acc_rate = self.reject_when_full()

    def produce_env_list(self):
        env_list = []
        for i in range(self.num_episode):
            env_list.append(Environment(total_bandwidth = 10,\
            distribution_list=[self.dist1,self.dist2,self.dist3], \
            mu_list=[1,2,3], lambda_list=[3,2,1],\
            num_of_each_type_distribution_list=[300,300,300]))
        return env_list

    def evaluate(self, agent):
        eval_reward = []
        accept_rate = []
        for i in range(self.num_episode):
            env = self.env_list[i]
            obs = env.reset()
            episode_reward = 0
            accept_num = 0
            count = 0
            while True:
                action = agent.predict(obs)
                if action == 1:
                    accept_num += 1
                obs, reward, done = env.step(action)
                episode_reward += reward
                if done:
                    break
                count += 1
            eval_reward.append(episode_reward/count)
            accept_rate.append(accept_num/count)
        return np.mean(eval_reward), np.mean(accept_rate)
    
    def evaluate_model(self, model):
        eval_reward = []
        accept_rate = []
        for i in range(self.num_episode):
            env = self.env_list[i]
            obs = env.reset()
            episode_reward = 0
            accept_num = 0
            count = 0
            while True:
                obs = torch.from_numpy(obs).view(1,-1)
                action = model(obs).argmax().item()
                if action == 1:
                    accept_num += 1
                obs, reward, done = env.step(action)
                episode_reward += reward
                if done:
                    break
                count += 1
            eval_reward.append(episode_reward/count)
            accept_rate.append(accept_num/count)
        return np.mean(eval_reward), np.mean(accept_rate)

    def reject_when_full(self):
        eval_reward = []
        accept_rate = []
        for i in range(self.num_episode):
            env = self.env_list[i]
            
            obs = env.reset()
            
            episode_reward = 0
            accept_num = 0
            count = 0
            while True:
                if obs[0]-obs[1]>=0:
                    action=1
                else:
                    action=0
                if action == 1:
                    accept_num += 1
                obs, reward, done = env.step(action)
                episode_reward += reward
                if done:
                    break
                count += 1
            eval_reward.append(episode_reward/count)
            accept_rate.append(accept_num/count)
            
        return np.mean(eval_reward), np.mean(accept_rate)

    def always_reject(self):
        eval_reward = []
        accept_rate = []
        for i in range(self.num_episode):
            env = self.env_list[i]
            obs = env.reset()
            episode_reward = 0
            accept_num = 0
            count = 0
            while True:
                
                action = 0
                if action == 1:
                    accept_num += 1
                obs, reward, done = env.step(action)
                episode_reward += reward
                if done:
                    break
                count += 1
            eval_reward.append(episode_reward/count)
            accept_rate.append(accept_num/count)
            
        return np.mean(eval_reward), np.mean(accept_rate)
    
    def always_accept(self):
        eval_reward = []
        accept_rate = []
        for i in range(self.num_episode):
            env = self.env_list[i]
            obs = env.reset()
            episode_reward = 0
            accept_num = 0
            count = 0
            while True:
                
                action = 1
                if action == 1:
                    accept_num += 1
                obs, reward, done = env.step(action)
                episode_reward += reward
                if done:
                    break
                count += 1
            eval_reward.append(episode_reward/count)
            accept_rate.append(accept_num/count)
           
        return np.mean(eval_reward), np.mean(accept_rate)
