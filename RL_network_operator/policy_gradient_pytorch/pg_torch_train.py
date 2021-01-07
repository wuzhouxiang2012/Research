import sys
import os
import torch
import numpy as np


sys.path.append("/Users/Bob/Documents/GitHub/Research/RL_network_operator/project")

from network_env import Environment
from distribution import Distribution
from request import Request

from pg_torch_model import PGTorchModel
from pg_torch_agent import PGTorchAgent
from pg_torch_algorithm import PGTorchAlgorithm

LEARNING_RATE = 1e-3


# 训练一个episode
def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        # obs, reward, done, info = env.step(action)
        obs, reward, done = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    num_accept = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        accept = 0
        while True:
            action = agent.predict(obs)
            if action==1:
                accept += 1
            # obs, reward, isOver, _ = env.step(action)
            obs, reward, isOver = env.step(action)
            episode_reward += reward
            # if render:
            #     env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
        num_accept.append(accept)
    return np.mean(eval_reward), np.mean(num_accept)


def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)


def main():
     # create environment
    dist1 = Distribution(id=0, vals=[2], probs=[1])
    dist2 = Distribution(id=1, vals=[5], probs=[1])
    dist3 = Distribution(id=2, vals=[2,8], probs=[0.5,0.5])

    env = Environment(total_bandwidth = 10,\
        distribution_list=[dist1,dist2,dist3], \
        mu_list=[1,2,3], lambda_list=[3,2,1],\
        num_of_each_type_distribution_list=[300,300,300])
    # env = gym.make('CartPole-v0')
    # env = env.unwrapped # Cancel the minimum score limit
    # obs_dim = env.observation_space.shape[0]
    # act_dim = env.action_space.n
    obs_dim = 6
    act_dim = 2
    # logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 根据parl框架构建agent
    model = PGTorchModel(state_dim=obs_dim, act_dim=act_dim)
    alg = PGTorchAlgorithm(model, lr=LEARNING_RATE)
    agent = PGTorchAgent(alg, obs_dim=obs_dim, act_dim=act_dim)

    # 加载模型
    # if os.path.exists('./policy_grad_model.ckpt'):
    #     agent.restore('./policy_grad_model.ckpt')
        # run_episode(env, agent, train_or_test='test', render=True)
        # exit()

    for i in range(1000):
        obs_list, action_list, reward_list = run_episode(env, agent)
        if i % 10 == 0:
            # logger.info("Episode {}, Reward Sum {}.".format(
            #     i, sum(reward_list)))
            print("Episode {}, Reward Sum {}.".format(i, sum(reward_list)))
        batch_obs = torch.from_numpy(np.array(obs_list))
        batch_action = torch.from_numpy(np.array(action_list))
        batch_reward = torch.from_numpy(calc_reward_to_go(reward_list, gamma=0.9))

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            total_reward, num_accept = evaluate(env, agent, render=True)
            print('Test reward: {}, num of accept: {}'.format(total_reward, num_accept))

    # save the parameters to ./pg_torch_model
    agent.save('./pg_torch_model')


if __name__ == '__main__':
    main()