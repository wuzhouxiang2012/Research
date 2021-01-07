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
        # # log_prob = layers.cross_entropy(act_prob, action) # 交叉熵
        # log_prob = layers.reduce_sum(
        #     -1.0 * layers.log(act_prob) * layers.one_hot(
        #         action, act_prob.shape[1]),
        #     dim=1)
        log_prob = (-1.0*torch.log(act_out) * nn.functional.one_hot(action)).sum(1)
        cost = log_prob * reward
        cost = cost.mean()
        cost.backward()
        # cost = log_prob * reward
        # cost = layers.reduce_mean(cost)

        # optimizer = fluid.optimizer.Adam(self.lr)
        # optimizer.minimize(cost)
        # return cost
        # print(act_prob.shape, action.shape)
        # criteria = nn.CrossEntropyLoss()
        # loss = criteria(act_prob, action)
        # loss = loss * reward
        # print('====loss', loss.shape)
        # softmax
        # exp_act_out=torch.exp(act_out)
        # sum_exp_act_prob = exp_act_out.sum(axis=1).view(-1,1)
        # act_prob = exp_act_out/sum_exp_act_prob
        # log_act_prob = torch.log(act_prob)
        # action_one_hot = nn.functional.one_hot(action)
        # loss = -1.0*((log_act_prob*action_one_hot).sum(axis=1)*reward).mean()
        # loss.backward()
        # print('====loss', loss, reward.sum())

        
        #optimize
        self.optimizer.step()
        self.optimizer.zero_grad()
        # return loss.item()


        