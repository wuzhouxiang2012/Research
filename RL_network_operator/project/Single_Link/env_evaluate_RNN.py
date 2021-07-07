
import numpy as np
import random
import heapq
from math import log2
import math
import copy
from request import Request
from request_type import ElasticRequestType, StaticRequestType

class EvaluateEnvironment:
    def __init__(self, total_bandwidth, request_type_list, \
            num_of_each_request_type_list,\
            base_rate=1, elastic_rate=1.1, static_rate=0.9,\
            elastic_history_buffer_length=100):

        assert len(request_type_list)==len(num_of_each_request_type_list)

        self.elastic_history_buffer_length = elastic_history_buffer_length
        self.request_type_list = request_type_list
        self.num_of_each_request_type_list = num_of_each_request_type_list
        self.total_bandwidth = total_bandwidth
        self.total_request_list = None
        self.current_request = None
        self.remaining_bandwidth = None
        self.accepted_request_heap = None

       # params for caculate reward
        self.base_rate=base_rate
        self.elastic_rate = elastic_rate
        self.static_rate = static_rate

        self.init_request()
        self.copy_total_list = self.total_request_list

    def reset(self):
        self.total_request_list = copy.deepcopy(self.copy_total_list)
        self.accepted_request_heap = []
        self.elastic_history_queue = []
        self.current_request = self.total_request_list.pop(0)
        self.remaining_bandwidth = self.total_bandwidth
        # elastic request history
        # save 100 entries for each type of elastic request
        self.elastic_history_dist = self.create_elastic_history_dist_map()
        self.elastic_hisotry_sum = self.create_elastic_history_sum_map()

        return self.produce_obs()
    def init_request(self):
        # clear request list memory
        self.total_request_list = []

        # produce all request
        for request_type, num_of_request in zip(self.request_type_list, self.num_of_each_request_type_list):
            self.total_request_list.extend(self.produce_normal_request(request_type, num_of_request))
        # produce all switch request
        self.total_request_list.extend(self.produce_switch_request())
        # sort according to request.arrival_stamp
        self.total_request_list.sort(key=lambda  x: x.arrival_stamp)
        
    def create_elastic_history_sum_map(self):
        map = {}
        for request_type in self.request_type_list:
            if not request_type.isStatic:
                key = request_type.id
                value = [0. for _ in request_type.bandwidth_list]
                map[key] = value
        return map    

    def create_elastic_history_dist_map(self):
        map = {}
        for request_type in self.request_type_list:
            if not request_type.isStatic:
                key = request_type.id
                value = [0. for _ in request_type.bandwidth_list]
                map[key] = value
        return map 
    
    def update_elastic_history(self):
        if not self.current_request.type.isStatic:
            if len(self.elastic_history_queue)>self.elastic_history_buffer_length:
                # pop first elastic in history:
                delete_request = self.elastic_history_queue.pop(0)
                delete_bd = delete_request.bandwidth
                if delete_bd != delete_request.type.bandwidth_list[0]:
                    delete_bd += delete_request.type.bandwidth_list[0]
                bd_idx = delete_request.type.bandwidth_list.index(delete_bd)
                self.elastic_hisotry_sum[delete_request.type.id][bd_idx]-=delete_request.service_time
                total = sum(self.elastic_hisotry_sum[delete_request.type.id])
                self.elastic_history_dist[delete_request.type.id]=[x/total for x in self.elastic_hisotry_sum[delete_request.type.id]]
            self.elastic_history_queue.append(self.current_request)
            current_bd = self.current_request.bandwidth
            if current_bd != self.current_request.type.bandwidth_list[0]:
                current_bd += self.current_request.type.bandwidth_list[0]
            bd_idx = self.current_request.type.bandwidth_list.index(current_bd)
            self.elastic_hisotry_sum[self.current_request.type.id][bd_idx]-=self.current_request.service_time
            total = sum(self.elastic_hisotry_sum[self.current_request.type.id])
            self.elastic_history_dist[self.current_request.type.id]=[x/total for x in self.elastic_hisotry_sum[self.current_request.type.id]]

    def kl_divergence(self, p, q):
        ret = 0
        for i in range(len(p)):
            if p[i]!=0:
                ret += p[i] * log2(p[i]/q[i])
        return ret

    def reward(self,action):
        if not self.current_request.isScale:
            if self.current_request.type.isStatic:
                if action==1:
                    return self.current_request.bandwidth* self.current_request.service_time*self.base_rate*self.static_rate
                else:
                    return 0
            else:
                if action==1:
                    return self.current_request.bandwidth* self.current_request.service_time*self.base_rate*self.elastic_rate
                else:
                    return 0
        else:
            if action==1:
                return self.current_request.bandwidth* self.current_request.service_time*self.base_rate*self.elastic_rate
            else:
                # calculate exp(-KL(d1, d2))
                cur_dist = self.elastic_history_dist[self.current_request.type.id]
                report_dist = self.current_request.type.distribution
                KL = self.kl_divergence(cur_dist,report_dist)
                return -1.0*math.exp(-KL)*self.current_request.bandwidth* self.current_request.service_time*self.base_rate*self.elastic_rate

    def step(self,action):
        # update elastic history sum and activity list    
        self.update_elastic_history()

        reward = self.reward(action)
        # if action is rejection and current request is the base request
        # remove all coming scale request
        if action == 0 and not self.current_request.isScale:
            new_total_request_list = []
            for request in self.total_request_list:
                if not request.id == self.current_request.id:
                    new_total_request_list.append(request)
            self.total_request_list = new_total_request_list

        # update state
        if action==1:
            if self.current_request.bandwidth>self.remaining_bandwidth:
                reward = -10*reward
            else:
                self.remaining_bandwidth -= self.current_request.bandwidth
                heapq.heappush(self.accepted_request_heap, self.current_request)

        if len(self.total_request_list)<=1:
            finished = True
            return self.produce_obs(), reward, finished, 'blabla'
        self.current_request = self.total_request_list.pop(0)
         # pop out all leaved request
        while len(self.accepted_request_heap)>0 and \
            self.accepted_request_heap[0].leave_stamp<self.current_request.arrival_stamp:
            old_request = heapq.heappop(self.accepted_request_heap)
            self.remaining_bandwidth += old_request.bandwidth
        finished = False 
        return self.produce_obs(), reward, finished, 'blabla'
    

    def produce_normal_request(self, request_type, num_of_request):
        bandwidth_list = []
        for i in range(num_of_request):
            if request_type.isStatic:
                bandwidth_list.append(request_type.bandwidth)
            else:
                bandwidth_list.append(request_type.bandwidth_list[0])
        arrvial_inverals = np.random.exponential(1./request_type.coming_rate, num_of_request)
        arrvial_time_stamp_list = list(np.cumsum(arrvial_inverals))
        service_interval_list = list(np.random.exponential(1./request_type.service_rate, num_of_request))
        request_list = []
        for i in range(num_of_request):
            request_list.append(Request(str(arrvial_time_stamp_list[i].item())+str(request_type.id), \
                arrival_stamp =arrvial_time_stamp_list[i], bandwidth=bandwidth_list[i], \
                service_time=service_interval_list[i], type=request_type))
        return request_list
    
    def produce_switch_request(self):
        switch_request_list = []
        for request in self.total_request_list:
            if request.type.isStatic:
                continue
            # 1 initial bandwidth
            # 1.1 random choose index of request type potential value list
            idx = np.random.choice(len(request.type.distribution), p=request.type.distribution)
            current_bd = request.type.bandwidth_list[idx]
            current_time = request.arrival_stamp
            while current_time < request.leave_stamp:
                # mu = mu from current_bd to next_bd
                bd_idx = request.type.bandwidth_list.index(current_bd)
                current_switch_mu = copy.deepcopy(request.type.switch_mu_matrix[bd_idx])
                current_switch_mu[bd_idx]=0
                sum_mu = sum(current_switch_mu)
                current_switch_mu = [x/sum_mu for x in current_switch_mu]
                next_bd_idx = np.random.choice(len(request.type.distribution), p = current_switch_mu)
                mu = request.type.switch_mu_matrix[bd_idx][next_bd_idx]
                period_time = np.random.exponential(1./mu)
                if current_time + period_time > request.leave_stamp:
                    period_time = request.leave_stamp-current_time
                diff = current_bd - request.type.bandwidth_list[0]
                if diff>0:                    
                    switch_request_list.append(Request(id= request.id, \
                        arrival_stamp =current_time, bandwidth=diff, service_time=period_time, type=request.type, isScale=True))
                current_time += period_time
                current_bd = request.type.bandwidth_list[next_bd_idx]
        return switch_request_list
    def produce_obs(self):
        part1 = np.zeros(shape=(1,7), dtype=np.float32)
        part1[0,0] = self.remaining_bandwidth
        part1[0,1] = self.current_request.bandwidth
        part1[0,2] = self.current_request.service_time
        if self.current_request.type.isStatic:
            part1[0,3] = 1
        elif not self.current_request.isScale:
            cur_dist = self.elastic_history_dist[self.current_request.type.id]
            report_dist = self.current_request.type.distribution
            KL = self.kl_divergence(cur_dist,report_dist)
            part1[0,4] = 1
            part1[0,6] = math.exp(-KL)
        else:
            cur_dist = self.elastic_history_dist[self.current_request.type.id]
            report_dist = self.current_request.type.distribution
            KL = self.kl_divergence(cur_dist,report_dist)
            part1[0,5] = 1
            part1[0,6] = math.exp(-KL)
        
        if len(self.accepted_request_heap)==0:
            part2 = np.zeros(shape=(1,5), dtype=np.float32)
        else:
            part2 = np.zeros(shape=(len(self.accepted_request_heap), 5))
            for i in range(len(self.accepted_request_heap)):
                request = self.accepted_request_heap[i]
                part2[i,0] = request.bandwidth
                part2[i,1] = request.leave_stamp - self.current_request.arrival_stamp
                if request.type.isStatic:
                    part2[i,2] = 1
                elif not request.isScale:
                    part2[i,3] = 1
                else:
                    part2[i,4] = 1
        return [part1, part2]
    # def produce_obs(self):
    #     part1 = np.zeros(shape=(1,6), dtype=np.float32)
    #     part1[0,0] = self.remaining_bandwidth
    #     part1[0,1] = self.current_request.bandwidth
    #     part1[0,2] = self.current_request.service_time
    #     if self.current_request.type.isStatic:
    #         part1[0,3] = 1
    #     elif not self.current_request.isScale:
    #         part1[0,4] = 1
    #     else:
    #         part1[0,5] = 1
        
    #     if len(self.accepted_request_heap)==0:
    #         part2 = np.zeros(shape=(1,5), dtype=np.float32)
    #     else:
    #         part2 = np.zeros(shape=(len(self.accepted_request_heap), 5))
    #         for i in range(len(self.accepted_request_heap)):
    #             request = self.accepted_request_heap[i]
    #             part2[i,0] = request.bandwidth
    #             part2[i,1] = request.leave_stamp - self.current_request.arrival_stamp
    #             if request.type.isStatic:
    #                 part2[i,2] = 1
    #             elif not request.isScale:
    #                 part2[i,3] = 1
    #             else:
    #                 part2[i,4] = 1
    #     return [part1, part2]