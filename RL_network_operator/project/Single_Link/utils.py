import numpy as np
np.set_printoptions(precision=3)
import torch
import random
import copy
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
        obs, reward, done, _ = env.step(action)
        reward_list.append(reward)
        if done:
            break
    return obs_list, action_list, reward_list

def reject_when_full_RNN(env_list):
    eval_reward = []
    accept_rate = []
    accpet_static_rate = []
    accept_initial_rate = []
    accept_scale_rate = []
    for env in env_list:
        
        obs = env.reset()
        
        episode_reward = 0
        accept_num = 0
        accept_num_static =0
        accept_num_initial = 0
        accept_num_scale = 0
        count = 0
        count_static = 0
        count_initial = 0
        count_scale = 1
        while True:
            count += 1
            if obs[0][0,3] == 1:
                count_static += 1
            if obs[0][0,4] == 1:
                count_initial += 1
            if obs[0][0,5] == 1:
                count_scale += 1
            if obs[0][0,0]-obs[0][0,1]>=0:
                action=1
            else:
                action=0
            if action == 1:
                accept_num += 1
                if obs[0][0,3] == 1:
                    accept_num_static += 1
                if obs[0][0,4] == 1:
                    accept_num_initial += 1
                if obs[0][0,5] == 1:
                    accept_num_scale += 1
            obs, reward, done,_ = env.step(action)
            episode_reward += reward
            if done:
                break
        eval_reward.append(episode_reward)
        accept_rate.append(accept_num/count)
        accpet_static_rate.append(accept_num_static/count_static)
        accept_initial_rate.append(accept_num_initial/count_initial)
        accept_scale_rate.append(accept_num_scale/count_scale)
    avg_reward = np.mean(eval_reward).item()
    avg_acc = np.mean(accept_rate).item()
    avg_static = np.mean(accpet_static_rate).item()
    avg_initial = np.mean(accept_initial_rate).item()
    avg_scale = np.mean(accept_scale_rate).item()
    print(f'reject when full RNN model: average reward:{avg_reward:.3f},\n\t average acc rate:{avg_acc:.3f},\n\t average static request acc rate:{avg_static:.3f},\n\t average initial elastic request acc rate:{avg_initial:.3f},\n\t average scale request acc rate:{avg_scale:.3f}')    
    return avg_reward, avg_acc, avg_static, avg_initial, avg_scale


def reject_when_full(env_list):
    eval_reward = []
    accept_rate = []
    for env in env_list:
        
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
            obs, reward, done,_ = env.step(action)
            episode_reward += reward
            if done:
                break
            count += 1
        eval_reward.append(episode_reward)
        accept_rate.append(accept_num/count)
    
    avg_reward = np.mean(eval_reward).item()
    avg_acc = np.mean(accept_rate)
    print(f'reject when full: average reward:{avg_reward:.3f}, average acc rate:{avg_acc:.3f}')
    return avg_reward, avg_acc
def evaluate_RNN(env_list, agent):
    eval_reward = []
    accept_rate = []
    accpet_static_rate = []
    accept_initial_rate = []
    accept_scale_rate = []
    for env in env_list:
        
        obs = env.reset()
        
        episode_reward = 0
        accept_num = 0
        accept_num_static =0
        accept_num_initial = 0
        accept_num_scale = 0
        count = 0
        count_static = 0
        count_initial = 0
        count_scale = 1
        while True:
            count += 1
            if obs[0][0,3] == 1:
                count_static += 1
            if obs[0][0,4] == 1:
                count_initial += 1
            if obs[0][0,5] == 1:
                count_scale += 1
            action = agent.predict(obs)  # pick best action
            if action == 1:
                accept_num += 1
                if obs[0][0,3] == 1:
                    accept_num_static += 1
                if obs[0][0,4] == 1:
                    accept_num_initial += 1
                if obs[0][0,5] == 1:
                    accept_num_scale += 1
            obs, reward, done,_ = env.step(action)
            episode_reward += reward
            if done:
                break
        eval_reward.append(episode_reward)
        accept_rate.append(accept_num/count)
        accpet_static_rate.append(accept_num_static/count_static)
        accept_initial_rate.append(accept_num_initial/count_initial)
        accept_scale_rate.append(accept_num_scale/count_scale)
    avg_reward = np.mean(eval_reward).item()
    avg_acc = np.mean(accept_rate).item()
    avg_static = np.mean(accpet_static_rate).item()
    avg_initial = np.mean(accept_initial_rate).item()
    avg_scale = np.mean(accept_scale_rate).item()
    print(f'evaluate RNN model: average reward:{avg_reward:.3f},\n\t average acc rate:{avg_acc:.3f},\n\t average static request acc rate:{avg_static:.3f},\n\t average initial elastic request acc rate:{avg_initial:.3f},\n\t average scale request acc rate:{avg_scale:.3f}')    
    return avg_reward, avg_acc, avg_static, avg_initial, avg_scale

# evaluate agent, run 5 episodes, return mean reward
def evaluate(env_list, agent, render=False):
    eval_reward = []
    accept_rate = []
    for env in env_list:
        obs = env.reset()
        episode_reward = 0
        accept_num = 0
        count = 0
        while True:
            action = agent.predict(obs)  # pick best action
            if action == 1:
                accept_num += 1
            obs, reward, done,_ = env.step(action)
            episode_reward += reward
            if done:
                break
            count += 1
        eval_reward.append(episode_reward)
        accept_rate.append(accept_num/count)
    avg_reward = np.mean(eval_reward).item()
    avg_acc = np.mean(accept_rate)
    print(f'evaluate : average reward:{avg_reward:.3f}, average acc rate:{avg_acc:.3f}')
    return avg_reward, avg_acc


def always_accept(env_list):
    eval_reward = []
    accept_rate = []
    for env in env_list:
        
        obs = env.reset()
        
        episode_reward = 0
        accept_num = 0
        count = 0
        while True:
            action=1
            if action == 1:
                accept_num += 1
            obs, reward, done,_ = env.step(action)
            episode_reward += reward
            count += 1
            if done:
                break
        eval_reward.append(episode_reward)
        accept_rate.append(accept_num/count)
    avg_reward = np.mean(eval_reward).item()
    avg_acc = np.mean(accept_rate)
    print(f'always accept: average reward:{avg_reward:.3f}, average acc rate:{avg_acc:.3f}')
    return avg_reward, avg_acc

def always_reject(env_list):
    eval_reward = []
    accept_rate = []
    for env in env_list:
        
        obs = env.reset()
        
        episode_reward = 0
        accept_num = 0
        count = 0
        while True:
            action=0
            if action == 1:
                accept_num += 1
            obs, reward, done,_ = env.step(action)
            episode_reward += reward
            count += 1
            if done:
                break
        eval_reward.append(episode_reward)
        accept_rate.append(accept_num/count)
    avg_reward = np.mean(eval_reward).item()
    avg_acc = np.mean(accept_rate)
    print(f'always reject: average reward:{avg_reward:.3f}, average acc rate:{avg_acc:.3f}')
    return avg_reward, avg_acc

def random_decide(env_list):
    eval_reward = []
    accept_rate = []
    for env in env_list:
        
        obs = env.reset()
        
        episode_reward = 0
        accept_num = 0
        count = 0
        while True:
            if random.random()>0.5:
                action=0
            else:
                action=1
                accept_num += 1
            obs, reward, done,_ = env.step(action)
            episode_reward += reward
            count += 1
            if done:
                break
        eval_reward.append(episode_reward)
        accept_rate.append(accept_num/count)
    avg_reward = np.mean(eval_reward).item()
    avg_acc = np.mean(accept_rate)
    print(f'random choice: average reward:{avg_reward:.3f}, average acc rate:{avg_acc:.3f}')
    return avg_reward, avg_acc

def random_decide_RNN(env_list, acc_rate=0.5):
    eval_reward = []
    accept_rate = []
    accpet_static_rate = []
    accept_initial_rate = []
    accept_scale_rate = []
    for env in env_list:
        
        obs = env.reset()
        
        episode_reward = 0
        accept_num = 0
        accept_num_static =0
        accept_num_initial = 0
        accept_num_scale = 0
        count = 0
        count_static = 0
        count_initial = 0
        count_scale = 1
        while True:
            count += 1
            if obs[0][0,3] == 1:
                count_static += 1
            if obs[0][0,4] == 1:
                count_initial += 1
            if obs[0][0,5] == 1:
                count_scale += 1
            if obs[0][0,0]-obs[0][0,1]>=0:
                if random.random()>acc_rate:
                    action=0
                else:
                    action=1
            else:
                action=0
            if action == 1:
                accept_num += 1
                if obs[0][0,3] == 1:
                    accept_num_static += 1
                if obs[0][0,4] == 1:
                    accept_num_initial += 1
                if obs[0][0,5] == 1:
                    accept_num_scale += 1
            obs, reward, done,_ = env.step(action)
            episode_reward += reward
            if done:
                break
        eval_reward.append(episode_reward)
        accept_rate.append(accept_num/count)
        accpet_static_rate.append(accept_num_static/count_static)
        accept_initial_rate.append(accept_num_initial/count_initial)
        accept_scale_rate.append(accept_num_scale/count_scale)
    avg_reward = np.mean(eval_reward).item()
    avg_acc = np.mean(accept_rate).item()
    avg_static = np.mean(accpet_static_rate).item()
    avg_initial = np.mean(accept_initial_rate).item()
    avg_scale = np.mean(accept_scale_rate).item()
    print(f'random decide: average reward:{avg_reward:.3f},\n\t average acc rate:{avg_acc:.3f},\n\t average static request acc rate:{avg_static:.3f},\n\t average initial elastic request acc rate:{avg_initial:.3f},\n\t average scale request acc rate:{avg_scale:.3f}')    
    return avg_reward, avg_acc, avg_static, avg_initial, avg_scale

def dfs(env):
    env_copy0 = copy.deepcopy(env)
    env_copy1 = copy.deepcopy(env)
    state0, reward0, finished0, _ = env_copy0.step(0)
    if state0[0][0,0]-state0[0][0,1]<0:
        return -100000
    state1, reward1, finished1, _ = env_copy1.step(1)
    if state1[0][0,0]-state1[0][0,1]<0:
        return reward0+dfs(env_copy0)
    ret0 = reward0
    ret1 = reward1
    if not finished0:
        ret0 += dfs(env_copy0)
    if not finished1:
        ret1 += dfs(env_copy1)
    return max(ret0, ret1)
    
def find_optimal(env):
    env.reset()

    return dfs(env)


def random_decide_for_rest_bandwidth(env_list):
    eval_reward = []
    accept_rate = []
    accpet_static_rate = []
    accept_initial_rate = []
    accept_scale_rate = []
    for env in env_list:
        
        obs = env.reset()
        
        episode_reward = 0
        accept_num = 0
        accept_num_static =0
        accept_num_initial = 0
        accept_num_scale = 0
        count = 0
        count_static = 0
        count_initial = 0
        count_scale = 1
        while True:
            count += 1
            if obs[0][0,3] == 1:
                count_static += 1
            if obs[0][0,4] == 1:
                count_initial += 1
            if obs[0][0,5] == 1:
                count_scale += 1
            if obs[0][0,0]-obs[0][0,1]>=0:
                acc_rate = (obs[0][0,0]-obs[0][0,1])/env.total_bandwidth
                if random.random()>acc_rate:
                    action=0
                else:
                    action=1
            else:
                action=0
            if action == 1:
                accept_num += 1
                if obs[0][0,3] == 1:
                    accept_num_static += 1
                if obs[0][0,4] == 1:
                    accept_num_initial += 1
                if obs[0][0,5] == 1:
                    accept_num_scale += 1
            obs, reward, done,_ = env.step(action)
            episode_reward += reward
            if done:
                break
        eval_reward.append(episode_reward)
        accept_rate.append(accept_num/count)
        accpet_static_rate.append(accept_num_static/count_static)
        accept_initial_rate.append(accept_num_initial/count_initial)
        accept_scale_rate.append(accept_num_scale/count_scale)
    avg_reward = np.mean(eval_reward).item()
    avg_acc = np.mean(accept_rate).item()
    avg_static = np.mean(accpet_static_rate).item()
    avg_initial = np.mean(accept_initial_rate).item()
    avg_scale = np.mean(accept_scale_rate).item()
    print(f'random decide: average reward:{avg_reward:.3f},\n\t average acc rate:{avg_acc:.3f},\n\t average static request acc rate:{avg_static:.3f},\n\t average initial elastic request acc rate:{avg_initial:.3f},\n\t average scale request acc rate:{avg_scale:.3f}')    
    return avg_reward, avg_acc, avg_static, avg_initial, avg_scale

# import pickle
# test_env_dir = open('20-300-5_test_env_list.obj', 'rb')
# test_env_list = pickle.load(test_env_dir)
# print("accept_rate depends on residual bandwidth/total")
# random_decide_for_rest_bandwidth(test_env_list)
# print('accept_rate = 0.5')
# random_decide_RNN(test_env_list)
# print("accept_rate = 0.8")
# random_decide_RNN(test_env_list, acc_rate=0.8)
# reject_when_full_RNN(test_env_list)
