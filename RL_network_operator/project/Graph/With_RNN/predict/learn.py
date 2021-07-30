import sys
sys.path.append(".")
import torch

from environment import Env_generator
env = Env_generator.produce_env(total_time=600, punish_flag=True, valid_flag=True)
print(len(env.total_request_list))

