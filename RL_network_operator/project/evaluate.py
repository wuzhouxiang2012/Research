import numpy as np
import os
from network_env import Environment
from distribution import Distribution
from request import Request

import sys
sys.path.append("/Users/Bob/Documents/GitHub/Research/RL_network_operator/policy_gradient")
# print(sys.path)
from model import Model
from agent import Agent
from algorithm import PolicyGradient

def always_accept(state):
    return 1
def always_reject(state):
    return 0
def reject_when_full(state):
    if state[0]-state[1]>=0:
        return 1
    else:
        return 0

def policy_gradient(state):
    obs_dim = 6
    act_dim = 2
    # logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 根据parl框架构建agent
    model = Model(act_dim=act_dim)
    alg = PolicyGradient(model, lr=0.01)
    agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)

    # 加载模型
    if os.path.exists('./policy_grad_model.ckpt'):
        agent.restore('./policy_grad_model.ckpt')
    action = agent.predict(state)
    return action
# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, model, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = model(obs)  # 预测动作，只选最优动作
            # obs, reward, done, _ = env.step(action)
            obs, reward, done = env.step(action)
            episode_reward += reward
            # if render:
            #     env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

def main():
     # create environment
    dist1 = Distribution(id=0, vals=[2], probs=[1])
    dist2 = Distribution(id=1, vals=[5], probs=[1])
    dist3 = Distribution(id=2, vals=[2,8], probs=[0.5,0.5])

    env = Environment(total_bandwidth = 10,\
        distribution_list=[dist1,dist2,dist3], \
        mu_list=[1,2,3], lambda_list=[3,2,1],\
        num_of_each_type_distribution_list=[300,300,300])

    
    print('always accept reward', evaluate(env, always_accept))
    print('always reject reward', evaluate(env, always_reject))
    print('reject when full reward', evaluate(env, reject_when_full))
    print('policy_gradient reward', evaluate(env, policy_gradient))
if __name__ == '__main__':
    main()