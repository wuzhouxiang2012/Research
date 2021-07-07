from numpy.core.fromnumeric import prod
from PPO import PPO2
from Env_generator import produce_env
env = produce_env()
ppo2 = PPO2(env, )