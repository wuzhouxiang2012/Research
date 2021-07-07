from torch.utils.data import DataLoader
from Util import evaluate
from actor import Actor
from agent import Agent
from dataset import Dataset
class PPO2():

    def __init__(self, env, evaluate_env_path:str, agent:Agent, actor:Actor, \
        lr:float, epsilon:float, gamm:float, base_line:float,\
        num_iter:int, num_episode:int, num_epoch:int) -> None:
        self.lr = lr
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.num_episode = num_episode
        self.num_epoch = num_epoch
        self.gamm = gamm
        self.base_line = base_line
        self.agent = agent
        self.actor = actor
        self.env = env
        self.evaluate_env_path = evaluate_env_path
        
    

    def calc_advantage(self, reward_list, gamma=1.0):
        for i in range(len(reward_list) - 2, -1, -1):
            # G_i = r_i + gamma * G_i+1
            reward_list[i] += gamma * reward_list[i + 1] - self.base_line # Gt
        return reward_list
        
    # run episode for train
    def run_episode(self, agent):
        obs_list, action_list, reward_list = [], [], []
        obs = self.env.reset()
        while True:
            obs_list.append(obs)
            # choose action based on prob
            action = agent.sample(obs)
            action_list.append(action)
            next_obs, reward, done, _ = self.env.step(action)
            reward_list.append(reward)
            if done:
                break
            obs = next_obs
        return obs_list, action_list, reward_list


    def train(self, env, evaluate_env_path, gamma = 0.9):
        env = self.env
        action_dim = env.action_space.n 
        obs_dim = env.observation_space.shape[0] 
        PPOactor = Actor(obs_size=obs_dim, action_size=action_dim)
        agent = Agent(
            actor=PPOactor,
            obs_dim = obs_dim,
            action_dim=action_dim)

        for iter in range(self.num_iter):
            #2.1  Using theta k to interact with the env
            # to collect {s_t, a_t} and compute advantage
            # advantage(s_t, a_t) = sum_{t^prime=t}^{T_n}(r_{t^prime}^{n})
            
            all_obs = []
            all_action = []
            all_advantage = []
            for episode in range(self.num_episode):
                obs_list, action_list, reward_list = self.run_episode(env, agent)
                advantage_list = self.calc_advantage(reward_list, gamma=gamma)
                all_obs.extend(obs_list)
                all_action.extend(action_list)
                all_advantage.extend(advantage_list)
            dataset = Dataset(obs_list=all_obs, action_list=all_action, advantage_list=all_advantage)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

            # optimize theta
            
            for epoch in range(self.num_epoch):
                for i, (batch_obs, batch_action, batch_adv) in enumerate(dataloader):
                    agent.learn(batch_obs, batch_action, batch_adv)

            if iter%10 == 0:
                eval_reward= evaluate(evaluate_env_path, agent, render=False)  # render=True 查看显示效果
                print('itern:{}  Test reward:{}'.format(iter, eval_reward))





