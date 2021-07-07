import copy
import torch
from torch import nn
class DQNPytorchAlg():
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ DQN algorithm
        
        Args:
            model (parl.Model): 定义Q函数的前向网络结构
            act_dim (int): action空间的维度，即有几个action
            gamma (float): reward的衰减因子
            lr (float): learning_rate，学习率.
        """
        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

        self.model = model
        self.target_model = copy.deepcopy(model)
        self.criteria = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    
    def predict(self, obs):
        with torch.no_grad():
            action = self.model(obs)
            return action

    def learn(self, obs_list, action_list, reward_list, next_obs_list):
        """ 使用DQN算法更新self.model的value网络
        """

        # 从target_model中获取 max Q' 的值，用于计算target_Q
        next_pred_value = self.target_model(next_obs_list)
        with torch.no_grad(): 
            target_value = reward_list + self.gamma*self.target_model(next_obs_list).max(1)[0]


        pred_value = self.model(obs_list)  # 获取Q预测值

        # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        action_onehot = torch.nn.functional.one_hot(action_list, self.act_dim)
        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        #  ==> pred_action_value = [[3.9]]
        # pred_action_value = layers.reduce_sum(
        #     layers.elementwise_mul(action_onehot, pred_value), dim=1)
        pred_value = (action_onehot*pred_value).sum(axis=1)
        loss = self.criteria(pred_value, target_value)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        # cost = layers.square_error_cost(pred_action_value, target)
        # cost = layers.reduce_mean(cost)
        # optimizer = fluid.optimizer.Adam(learning_rate=self.lr)  # 使用Adam优化器
        # optimizer.minimize(cost)
        # return cost


    def sync_target(self):
        """ 把 self.model 的模型参数值同步到 self.target_model
        """
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

