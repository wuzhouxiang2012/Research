
import numpy as np
import random
import heapq
from math import log2
import math
from request import Request
from distribution import Distribution
import torch

class Environment:
    '''
    distribution_id_list: save all distribution type
    mu_list: save all corresponding service time param mu
    lambda_list: save all corresponding interval time param mu
    '''
    def __init__(self, total_bandwidth, distribution_list, \
            mu_list, lambda_list, num_of_each_type_distribution_list,\
            base_rate=1, elastic_rate=1.1, static_rate=0.9,\
            elastic_history_buffer_length=100):

        assert len(distribution_list)==len(mu_list)
        assert len(mu_list) == len(lambda_list)
        assert len(lambda_list) == len(num_of_each_type_distribution_list)

        self.distribution_list = distribution_list
        self.mu_list = mu_list
        self.lambda_list = lambda_list
        self.num_of_each_type_distribution_list = num_of_each_type_distribution_list
        self.total_bandwidth = total_bandwidth
        self.total_request_list = None
        self.current_request_idx = None
        self.current_request = None
        self.remaining_bandwidth = None
        self.accepted_request_heap = None

        self.elastic_history_buffer_length = elastic_history_buffer_length

        # params for caculate reward
        self.base_rate=base_rate
        self.elastic_rate = elastic_rate
        self.static_rate = static_rate

        # distribution id one hot => diagonal mat
        self.distribution_one_hot = np.diag([1 for _ in range(len(self.distribution_list))])

        self.init_request()

    def reset(self):
        
        self.accepted_request_heap = []
        self.current_request_idx = 0
        self.current_request = self.total_request_list[self.current_request_idx]
        self.remaining_bandwidth = self.total_bandwidth
        # elastic request history
        # save 100 entries for each type of elastic request
        self.elastic_history_activity = self.create_elastic_history_acitivty_map()
        self.elastic_hisotry_sum = self.create_elastic_history_sum_map()

        return self.produce_obs()
    def init_request(self):
        # clear request list memory
        self.total_request_list = []

        # produce all request
        for request_dist, num_of_request, lamb, mu in zip(self.distribution_list, self.num_of_each_type_distribution_list, self.lambda_list, self.mu_list):
            self.total_request_list.extend(self.produce_request(request_dist, num_of_request, lamb, mu))
        # sort according to request.arrival_stamp
        self.total_request_list.sort(key=lambda  x: x.arrival_stamp)
        
        


    def produce_obs(self):
        obs = []
        obs.append(self.remaining_bandwidth)
        obs.append(self.current_request.bandwidth)
        obs.append(self.current_request.service_time)
        obs.extend(list(self.distribution_one_hot[self.current_request.distribution.id,:]))
        return np.array(obs, dtype=np.float32)

    def step(self,action):
        # print(self.current_request)
        # pop out all leaved request
        while len(self.accepted_request_heap)>0 and \
            self.accepted_request_heap[0].leave_stamp<self.current_request.arrival_stamp:
            old_request = heapq.heappop(self.accepted_request_heap)
            self.remaining_bandwidth += old_request.bandwidth

        # update elastic history sum and activity list
        if not self.current_request.distribution.isStatic:
            self.update_elastic_history()

        reward = self.reward(action)

        # update index
        self.current_request_idx += 1
        self.current_request = self.total_request_list[self.current_request_idx]

        finished = False 
        # discard final step
        if self.current_request_idx==len(self.total_request_list)-1:
            finished=True

        return self.produce_obs(), reward, finished
    
    def reward(self, action):
        # calculate reward
        reward = 0.0
        if action==1 and self.current_request.distribution.isStatic:
            reward = self.current_request.bandwidth*self.base_rate*self.static_rate*\
                self.current_request.service_time
            if self.remaining_bandwidth<self.current_request.bandwidth:
                reward = -reward*10
                # reward = -reward
            else:
                heapq.heappush(self.accepted_request_heap, self.current_request)
                self.remaining_bandwidth -= self.current_request.bandwidth
        elif action==1 and not self.current_request.distribution.isStatic:
            reward = self.current_request.bandwidth*self.base_rate*self.elastic_rate*\
                self.current_request.service_time
            if self.remaining_bandwidth<self.current_request.bandwidth:
                reward = -reward*10
                # reward = -reward
            else:
                heapq.heappush(self.accepted_request_heap, self.current_request)
                self.remaining_bandwidth -= self.current_request.bandwidth
        elif action==0 and self.current_request.distribution.isStatic:
            reward = -1*self.current_request.bandwidth*self.base_rate*self.static_rate*\
                self.current_request.service_time
        elif action==0 and not self.current_request.distribution.isStatic:
            cur_dist = self.cal_current_distribution(self.elastic_hisotry_sum, \
                self.current_request.distribution.id)

            kl_div = self.kl_divergence(cur_dist, \
                self.current_request.distribution.probs)
            average_history = self.cal_average(self.current_request.distribution.vals, \
                cur_dist)
            average_reported = self.cal_average(self.current_request.distribution.vals, \
                self.current_request.distribution.probs)
            
            # E[history]>E[reported distribution] -> punish less according to kl
            if average_history > average_reported:
                reward = -1 * self.current_request.bandwidth*self.base_rate*self.elastic_rate*\
                self.current_request.service_time * math.exp(-kl_div)
            else:
                reward = -1 * self.current_request.bandwidth*self.base_rate*self.elastic_rate*\
                self.current_request.service_time
        return reward

    def create_elastic_history_acitivty_map(self):
        history_map = {}
        for dist in self.distribution_list:
            if not dist.isStatic:
                history_map[dist.id] = []
        return history_map

    def create_elastic_history_sum_map(self):
        history_sum_map = {}
        for dist in self.distribution_list:
            if not dist.isStatic:
                history_sum_map[dist.id] = [0 for _ in range(len(dist.probs))]
        return history_sum_map
    
    def update_elastic_history(self):
        # unwrap request
        request_bandwidth = self.current_request.bandwidth
        service_time = self.current_request.service_time
        distribution = self.current_request.distribution
        idx_in_distribution = distribution.vals.index(request_bandwidth)   
        self.elastic_hisotry_sum[distribution.id][idx_in_distribution] += service_time
        # update history list
        self.elastic_history_activity[distribution.id].append(self.current_request)
        if len(self.elastic_history_activity[distribution.id]) > self.elastic_history_buffer_length:
            pop_request = self.elastic_history_activity[distribution.id].pop(0)
            request_bandwidth = pop_request.bandwidth
            service_time = pop_request.service_time
            distribution = pop_request.distribution
            idx_in_distribution = distribution.vals.index(request_bandwidth)
            self.elastic_hisotry_sum[distribution.id][idx_in_distribution] -= service_time
    
    def kl_divergence(self, p, q):
        ret = 0
        for i in range(len(p)):
            if p[i]!=0:
                ret += p[i] * log2(p[i]/q[i])
        return ret
        
    def cal_current_distribution(self, history_sum, id):
        total = sum(history_sum[id])
        return [x/total for x in history_sum[id]]
    def cal_average(self, vals, dist):
        return sum([x*y for x,y in zip(vals,dist)])

    def produce_request(self, request_dist, num_of_request, lamb, mu):
        bandwidth_list = []
        for i in range(num_of_request):
            if request_dist.isStatic:
                bandwidth_list.append(request_dist.vals[0])
            else:
                rand = random.random()
                idx = 0
                while rand>request_dist.probs[idx]:
                    rand -= request_dist.probs[idx]
                    idx+=1 
                bandwidth_list.append(request_dist.vals[idx])


        arrvial_inverals = np.random.exponential(lamb, num_of_request)
        arrvial_time_stamp_list = list(np.cumsum(arrvial_inverals))
        service_interval_list = list(np.random.exponential(mu, num_of_request))
        request_list = []
        for i in range(num_of_request):
            request_list.append(Request(arrival_stamp =arrvial_time_stamp_list[i], bandwidth=bandwidth_list[i], service_time=service_interval_list[i], distribution=request_dist))

        return request_list


# if __name__ == "__main__":
#     # create environment
#     dist1 = Distribution(id=0, vals=[2], probs=[1])
#     dist2 = Distribution(id=1, vals=[5], probs=[1])
#     dist3 = Distribution(id=2, vals=[2,8], probs=[0.5,0.5])

#     env = Environment(total_bandwidth = 10,\
#         distribution_list=[dist1,dist2,dist3], \
#         mu_list=[1,2,3], lambda_list=[3,2,1],\
#         num_of_each_type_distribution_list=[3,3,3])
    
#     state = env.reset()
#     i =0
#     r = 0
#     while True:
#         next_state, reward, finished = env.step(1)
#         r+= reward
#         print(i, state, reward)
#         i += 1
#         state = next_state
#         if finished:
#             break
#     print('accept', r)

#     state = env.reset()
#     i =0
#     r = 0
#     while True:
#         next_state, reward, finished = env.step(0)
#         r+= reward
#         print(i, state, reward)
#         i += 1
#         state = next_state
#         if finished:
#             break
#     print('reject', r)

#     state = env.reset()
#     i =0
#     r = 0
#     while True:
#         if state[0]>=state[1]:
#             action = 1
#         else:
#             action = 0
#         next_state, reward, finished = env.step(action)
#         r+= reward
#         print(i, state, reward)
#         i += 1
#         state = next_state
#         if finished:
#             break
#     print('basic', r)

