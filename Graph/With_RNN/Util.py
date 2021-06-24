import pickle
import random 
from math import log2
import numpy as np

def kl_divergence(p, q):
    ret = 0
    for i in range(len(p)):
        if p[i]!=0:
            ret += p[i] * log2(p[i]/q[i])
    return ret

def reject_when_full(env):
    total_reward = 0.0
    obs = env.reset()
    while True:
        choosed_action = 0
        for action in range(1,4):
            if env.valid_deploy(action=action):
                choosed_action = action
                break
        obs, reward, finished, _ = env.step(choosed_action)
        total_reward += reward
        if finished:
            break
    return total_reward

def totally_random(env):
    total_reward = 0.0
    obs = env.reset()
    while True:
        choosed_action = random.choice([0,1,2,3])
        obs, reward, finished, _ = env.step(choosed_action)
        total_reward += reward
        if finished:
            break
    return total_reward


def evaluate(env_list_path, agent, render=False):
    eval_reward = []
    filehandler = open(env_list_path,"rb")
    env_list = pickle.load(filehandler)
    filehandler.close()
    for env in env_list:
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

def evaluate_stable(env_list_path, agent, render=False):
    eval_reward = []
    filehandler = open(env_list_path,"rb")
    env_list = pickle.load(filehandler)
    filehandler.close()
    for env in env_list:
        obs = env.reset()
        episode_reward = 0
        while True:
            action, _state = agent.predict(obs)  # 预测动作，只选最优动作
            action = int(action)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

def evaluate_reject_when_full(env_list_path):
    eval_reward = []
    filehandler = open(env_list_path,"rb")
    env_list = pickle.load(filehandler)
    filehandler.close()
    for env in env_list:
        eval_reward.append(reject_when_full(env))
    return np.mean(eval_reward)

def evaluate_totally_random(env_list_path):
    eval_reward = []
    filehandler = open(env_list_path,"rb")
    env_list = pickle.load(filehandler)
    filehandler.close()
    for env in env_list:
        eval_reward.append(totally_random(env))
    return np.mean(eval_reward)


if __name__ == '__main__':
    print(evaluate_reject_when_full('env_list_set1'))