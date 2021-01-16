import numpy as np
import torch
def calc_advantage(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + gamma * G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return reward_list

# run episode for train
def run_episode(env, model):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs = torch.from_numpy(obs).view(1,-1)
        obs_list.append(obs)
        # choose action based on prob
        action = np.random.choice(range(2), p=model(obs).detach().numpy().reshape(-1,))  
        action_list.append(action)

        # obs, reward, done, info = env.step(action)
        obs, reward, done = env.step(action)
        reward_list.append(reward)
        if done:
            break
    return obs_list, action_list, reward_list
