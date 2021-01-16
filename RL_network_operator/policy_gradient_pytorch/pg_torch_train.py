import sys
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.append("/Users/Bob/Documents/GitHub/Research/RL_network_operator/project")

from network_env import Environment
from evaluate import Evaluation
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
def evaluate(env, agent):
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
    evaluation = Evaluation()
    obs_dim = 6
    act_dim = 2
    # logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 根据parl框架构建agent
    model = PGTorchModel(state_dim=obs_dim, act_dim=act_dim)
    alg = PGTorchAlgorithm(model, lr=LEARNING_RATE)
    agent = PGTorchAgent(alg, obs_dim=obs_dim, act_dim=act_dim)

    # 加载模型
    if os.path.exists('./pg_torch_model'):
        agent.restore('./pg_torch_model')
    writer = SummaryWriter()

    for i in range(1000):
        obs_list, action_list, reward_list = run_episode(env, agent)
        writer.add_scalars('Reward/train', {'train_reward':sum(reward_list)/len(reward_list), \
                'reject when full': evaluation.reject_when_full_avg_reward, \
                    'always accept': evaluation.always_accept_avg_reward,\
                        'always reject': evaluation.always_reject_avg_reward}, i)
        # writer.add_scalar('Reward/train', evaluation.always_reject_avg_reward, i)
        
        if i % 10 == 0:
            # logger.info("Episode {}, Reward Sum {}.".format(
            #     i, sum(reward_list)))
            print("Episode {}, Reward Sum {}.".format(i, sum(reward_list)/len(reward_list)))
        batch_obs = torch.from_numpy(np.array(obs_list))

        batch_action = torch.from_numpy(np.array(action_list))
        batch_reward = torch.from_numpy(calc_reward_to_go(reward_list, gamma=0.9))

        loss = agent.learn(batch_obs, batch_action, batch_reward)
        writer.add_scalar('Loss/train', loss, i)
        if (i + 1) % 100 == 0:
            avg_reward, avg_acc_rate = evaluation.evaluate(agent)
            writer.add_scalars('reward Test', {'test reward': avg_reward, \
                'reject when full': evaluation.reject_when_full_avg_reward, \
                    'always accept': evaluation.always_accept_avg_reward,\
                        'always reject': evaluation.always_reject_avg_reward}, i)
            writer.add_scalars('accept rate Test', {'test rate': avg_acc_rate, \
                'reject when full': evaluation.reject_when_full_avg_acc_rate, \
                    'always accept': evaluation.always_accept_avg_acc_rate,\
                        'always reject': evaluation.always_reject_avg_acc_rate}, i)
            print('avg_reward', avg_reward, 'avg_acc_rate', avg_acc_rate, 'base ', evaluation.reject_when_full_avg_reward)

    writer.close()
    # save the parameters to ./pg_torch_model
    agent.save('./pg_torch_model')


if __name__ == '__main__':
    main()