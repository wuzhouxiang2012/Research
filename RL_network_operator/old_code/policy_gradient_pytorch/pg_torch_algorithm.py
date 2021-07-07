import torch 
from torch import nn


class PGTorchAlgorithm():
    def __init__(self, model, lr=None):
        """ Policy Gradient algorithm
        
        Args:
            model (parl.Model): policy的前向网络.
            lr (float): 学习率.
        """

        self.model = model
        assert isinstance(lr, float)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def predict(self, obs):
        """ 使用policy model预测输出的动作概率
        """
        return self.model(obs)

    def learn(self, obs, action, reward):
        """ 用policy gradient 算法更新policy model
        """
        act_out = self.model(obs)  # 获取输出动作概率
        log_prob = (-1.0*torch.log(act_out) * nn.functional.one_hot(action)).sum(1)
        loss = log_prob * reward
        loss = loss.mean()
        loss.backward()
        #optimize
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()


        