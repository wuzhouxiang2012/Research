from Env_generator import produce_env
from Util import evaluate_stable
from stable_baselines3.common.env_checker import check_env
env = produce_env()
check_env(env)

from stable_baselines3 import A2C, DQN
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=500)
# print(evaluate_stable('env_list_set1', model))

model3 = DQN("MlpPolicy", env, verbose=1)
for _ in range(30):
    model3.learn(total_timesteps=250)
    print(evaluate_stable('env_list_set1', model3))