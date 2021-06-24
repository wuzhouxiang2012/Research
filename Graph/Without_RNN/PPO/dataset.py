import torch
import numpy as np
class Dataset(torch.utils.data.Dataset):
    def __init__(self, obs_list, action_list, advantage_list):
        self.obs_list = torch.from_numpy(np.array(obs_list).astype(np.float32))
        self.action_list = torch.tensor(action_list, dtype=torch.int64)
        self.advantage_list = torch.tensor(advantage_list, dtype=torch.float32)

    def __getitem__(self, index):
        return self.obs_list[index,:], self.action_list[index], self.advantage_list[index]
    
    def __len__(self):
        return self.obs_list.shape[0]