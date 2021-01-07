import numpy as np
import os
from network_env import Environment
from distribution import Distribution
from request import Request

class Evaluation():
    def __init__(self):
             # create environment
        dist1 = Distribution(id=0, vals=[2], probs=[1])
        dist2 = Distribution(id=1, vals=[5], probs=[1])
        dist3 = Distribution(id=2, vals=[2,8], probs=[0.5,0.5])

        self.env = Environment(total_bandwidth = 10,\
            distribution_list=[dist1,dist2,dist3], \
            mu_list=[1,2,3], lambda_list=[3,2,1],\
            num_of_each_type_distribution_list=[300,300,300])
        

    def evaluate(self, env, agent, render=False):
        eval_reward = []
        accept_rate = []
        for i in range(5):
            obs = self.env.reset()
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
            accept_rate.append(accept_rate/count)
        return np.mean(eval_reward), np.mean(accept_rate)
def always_accept(_):
    return 1
def always_reject(_):
    return 0
def reject_when_full(state):
    if state[0]-state[1]>=0:
        return 1
    else:
        return 0