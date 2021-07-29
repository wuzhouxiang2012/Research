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
def check_req_type(obs_):
    # print(obs_)
    # print(obs_.shape)
    if obs_[-1]==1:
        return 2
    if obs_[-2]==1:
        return 3
    if obs_[-3]==1:
        return 1
def reject_when_full(env):
    total_reward = 0.0
    obs = env.reset()
    obs = obs[0]
    total = 0 # total request
    accept = 0 # total accept
    total_sd = 0 # total static deployment type request
    accept_sd = 0 
    total_ed = 0 # total elastic deployment type request
    accept_ed = 0
    total_es = 0 # total elastic scaling type request
    accept_es = 0

    while True:
        total += 1
        if check_req_type(obs)==1:
            total_sd += 1
        if check_req_type(obs)==2:
            total_ed += 1
        if check_req_type(obs)==3:
            total_es += 1

        choosed_action = 0
        for action in range(1,4):
            if env.valid_deploy(action=action):
                choosed_action = action
                break
        if choosed_action != 0:
            accept+=1
            if check_req_type(obs)==1:
                accept_sd += 1
            if check_req_type(obs)==2:
                accept_ed += 1
            if check_req_type(obs)==3:
                accept_es += 1

        obs, reward, finished, _ = env.step(choosed_action)
        obs = obs[0]
        total_reward += reward
        if finished:
            break
    return total_reward, total, accept, \
        total_sd, accept_sd, total_ed, accept_ed, total_es, accept_es

def totally_random(env):
    total_reward = 0.0
    obs = env.reset()
    obs = obs[0]
    total = 0 # total request
    accept = 0 # total accept
    total_sd = 0 # total static deployment type request
    accept_sd = 0 
    total_ed = 0 # total elastic deployment type request
    accept_ed = 0
    total_es = 0 # total elastic scaling type request
    accept_es = 0

    while True:
        total += 1
        if check_req_type(obs)==1:
            total_sd += 1
        if check_req_type(obs)==2:
            total_ed += 1
        if check_req_type(obs)==3:
            total_es += 1

        choosed_action = random.choice([0,1,2,3])
        if choosed_action != 0:
            accept+=1
            if check_req_type(obs)==1:
                accept_sd += 1
            if check_req_type(obs)==2:
                accept_ed += 1
            if check_req_type(obs)==3:
                accept_es += 1

        obs, reward, finished, _ = env.step(choosed_action)
        obs = obs[0]
        total_reward += reward
        if finished:
            break
    return total_reward, total, accept, \
        total_sd, accept_sd, total_ed, accept_ed, total_es, accept_es

def _evaluate(env, agent):
    total_reward = 0.0
    obs = env.reset()
    total = 0 # total request
    accept = 0 # total accept
    total_sd = 0 # total static deployment type request
    accept_sd = 0 
    total_ed = 0 # total elastic deployment type request
    accept_ed = 0
    total_es = 0 # total elastic scaling type request
    accept_es = 0

    while True:
        total += 1
        if check_req_type(obs[0])==1:
            total_sd += 1
        if check_req_type(obs[0])==2:
            total_ed += 1
        if check_req_type(obs[0])==3:
            total_es += 1

        choosed_action = agent.predict(obs)
        
        if choosed_action != 0:
            accept+=1
            if check_req_type(obs[0])==1:
                accept_sd += 1
            if check_req_type(obs[0])==2:
                accept_ed += 1
            if check_req_type(obs[0])==3:
                accept_es += 1

        obs, reward, finished, _ = env.step(choosed_action)
        total_reward += reward
        if finished:
            break
    return total_reward, total, accept, \
        total_sd, accept_sd, total_ed, accept_ed, total_es, accept_es


def evaluate(env_list_path, agent, render=False):
    eval_reward_list = []
    total_req_list = []
    acc_req_list = []
    total_sd_list =[]
    acc_sd_list = []
    total_ed_list = []
    acc_ed_list = []
    total_es_list = []
    acc_es_list = []

    filehandler = open(env_list_path,"rb")
    env_list = pickle.load(filehandler)
    filehandler.close()
    for env in env_list:
        total_reward, total, accept, \
        total_sd, accept_sd, total_ed, \
        accept_ed, total_es, accept_es = _evaluate(env, agent)
        eval_reward_list.append(total_reward)
        total_req_list.append(total)
        acc_req_list.append(accept)
        total_sd_list.append(total_sd)
        acc_sd_list.append(accept_sd)
        total_ed_list.append(total_ed)
        acc_ed_list.append(accept_ed)
        total_es_list.append(total_es)
        acc_es_list.append(accept_es)

        mean_total_reward = np.mean(eval_reward_list)
        mean_total_accept = 0
        if np.sum(total_req_list)!= 0:
            mean_total_accept = np.sum(acc_req_list)/np.sum(total_req_list)
        mean_sd_accpet = 0
        if np.sum(total_sd_list) != 0:
            mean_sd_accpet = np.sum(acc_sd_list)/np.sum(total_sd_list)
        mean_ed_accpet = 0
        if np.sum(total_ed_list)!=0:
            
            mean_ed_accpet = np.sum(acc_ed_list)/np.sum(total_ed_list)
        mean_es_accept = 0
        if np.sum(total_es_list)!=0:
            mean_es_accept = np.sum(acc_es_list)/np.sum(total_es_list)
    return mean_total_reward, \
        mean_total_accept, \
        mean_sd_accpet, \
        mean_ed_accpet,\
        mean_es_accept

def evaluate_reject_when_full(env_list_path):
    eval_reward_list = []
    total_req_list = []
    acc_req_list = []
    total_sd_list =[]
    acc_sd_list = []
    total_ed_list = []
    acc_ed_list = []
    total_es_list = []
    acc_es_list = []

    filehandler = open(env_list_path,"rb")
    env_list = pickle.load(filehandler)
    filehandler.close()
    for env in env_list:
        total_reward, total, accept, \
        total_sd, accept_sd, total_ed, \
        accept_ed, total_es, accept_es = reject_when_full(env)
        eval_reward_list.append(total_reward)
        total_req_list.append(total)
        acc_req_list.append(accept)
        total_sd_list.append(total_sd)
        acc_sd_list.append(accept_sd)
        total_ed_list.append(total_ed)
        acc_ed_list.append(accept_ed)
        total_es_list.append(total_es)
        acc_es_list.append(accept_es)
    return np.mean(eval_reward_list), \
        np.sum(acc_req_list)/np.sum(total_req_list), \
        np.sum(acc_sd_list)/np.sum(total_sd_list), \
        np.sum(acc_ed_list)/np.sum(total_ed_list),\
        np.sum(acc_es_list)/np.sum(total_es_list)
        
def evaluate_totally_random(env_list_path):
    eval_reward_list = []
    total_req_list = []
    acc_req_list = []
    total_sd_list =[]
    acc_sd_list = []
    total_ed_list = []
    acc_ed_list = []
    total_es_list = []
    acc_es_list = []

    filehandler = open(env_list_path,"rb")
    env_list = pickle.load(filehandler)
    filehandler.close()
    for env in env_list:
        total_reward, total, accept, \
        total_sd, accept_sd, total_ed, \
        accept_ed, total_es, accept_es = totally_random(env)
        eval_reward_list.append(total_reward)
        total_req_list.append(total)
        acc_req_list.append(accept)
        total_sd_list.append(total_sd)
        acc_sd_list.append(accept_sd)
        total_ed_list.append(total_ed)
        acc_ed_list.append(accept_ed)
        total_es_list.append(total_es)
        acc_es_list.append(accept_es)
    return np.mean(eval_reward_list), \
        np.sum(acc_req_list)/np.sum(total_req_list), \
        np.sum(acc_sd_list)/np.sum(total_sd_list), \
        np.sum(acc_ed_list)/np.sum(total_ed_list),\
        np.sum(acc_es_list)/np.sum(total_es_list)


if __name__ == '__main__':
    print(evaluate_totally_random('env_list_set1'))