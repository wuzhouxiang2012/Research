import numpy as np
import torch
import sys
sys.path.append("/Users/Bob/Documents/GitHub/Research/RL_network_operator/project")

from network_env import Environment
from distribution import Distribution
from request import Request


from dqn_pytorch_model import DQNPtorchModel
from dqn_pytorch_alg import DQNPytorchAlg  # from parl.algorithms import DQN  # parl >= 1.3.1
from dqn_pytorch_agent import DQNPytorchAgent

from dqn_pytorch_replay_memory import DQNPytorchReplayMemory

LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001  # 学习率
GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等


# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        # next_obs, reward, done, _ = env.step(action)
        next_obs, reward, done = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    num_accpet = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        accept = 0
        while True:
            action = agent.predict(torch.from_numpy(obs))  # 预测动作，只选最优动作
            if action == 1:
                accept += 1
            obs, reward, done = env.step(action)
            episode_reward += reward
            # if render:
            #     env.render()
            if done:
                break
        eval_reward.append(episode_reward)
        num_accpet.append(accept)
    return np.mean(eval_reward), np.mean(num_accpet)


def main():
    dist1 = Distribution(id=0, vals=[2], probs=[1])
    dist2 = Distribution(id=1, vals=[5], probs=[1])
    dist3 = Distribution(id=2, vals=[2,8], probs=[0.5,0.5])

    env = Environment(total_bandwidth = 10,\
        distribution_list=[dist1,dist2,dist3], \
        mu_list=[1,2,3], lambda_list=[3,2,1],\
        num_of_each_type_distribution_list=[300,300,300])

    action_dim = 2
    state_dim = 6
    rpm = DQNPytorchReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

    # 根据parl框架构建agent
    model = DQNPtorchModel(state_dim=state_dim, act_dim=action_dim)
    algorithm = DQNPytorchAlg(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = DQNPytorchAgent(
        algorithm,
        obs_dim = state_dim,
        act_dim=action_dim,
        e_greed=0.1,  # 有一定概率随机选取动作，探索
        e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低

    # 加载模型
    # save_path = './dqn_model.ckpt'
    # agent.restore(save_path)

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)

    max_episode = 2000

    # start train
    episode = 0
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part
        for i in range(0, 50):
            total_reward = run_episode(env, agent, rpm)
            episode += 1

        # test part
        eval_reward, num_accpet = evaluate(env, agent)  # render=True 查看显示效果
        print(f'episode{episode}:evaluate reward,{eval_reward}, num of accpet:{num_accpet}')

    # 训练结束，保存模型
    save_path = './dqn_pytorch_model.ckpt'
    agent.save(save_path)


if __name__ == '__main__':
    main()
