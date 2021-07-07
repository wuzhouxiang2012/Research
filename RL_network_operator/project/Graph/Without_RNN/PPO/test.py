from numpy.core.fromnumeric import prod
from PPO import PPO2
from actor import Actor
from agent import Agent
from Env_generator import produce_env
env = produce_env()
actor = Actor()
agent = Agent()
ppo2 = PPO2(env, 'env_list_set1', )