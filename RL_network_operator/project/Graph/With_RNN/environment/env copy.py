import gym
from gym import spaces
from typing import Dict, List
import numpy as np
import random
import math
import heapq
import copy
from collections import OrderedDict
from request import Request
from request_type import RequestType
from util import kl_divergence

class Environment(gym.Env):
    def __init__(self, remain_dict:Dict[str, int],\
            edge_sequence:List[str],\
            node_num:int,\
            path_dict:Dict[str,Dict[int,List[str]]],\
            request_type_list:List[RequestType], \
            total_time:float,\
            base_rate:float=1, elastic_rate:float=1.1, \
            static_rate:float=0.9,\
            elastic_history_buffer_length:int=1000,\
            decrease_with_edge:float = 0.9,
            punish_flag:bool=True,
            valid_flag:bool=False):
        self.action_space = spaces.Discrete(4)
        LOW_BOUND = np.zeros((45,), dtype=np.float)
        HIGH_BOUND = np.repeat(np.finfo(np.float32).max, 45)
        self.observation_space = spaces.Box(low=LOW_BOUND, high=HIGH_BOUND, \
             dtype=np.float32)
        self.node_num = node_num
        # source one hot, sink one hot, bw, service time, type one hot
        self.req_encode_size = self.node_num*2 + 1 + 1 + 3
        self.elastic_history_buffer_length = elastic_history_buffer_length
        self.request_type_list = request_type_list
        self.total_time = total_time
        self.original_remain_dict = remain_dict
        
        self.edge_sequence = edge_sequence
        self.edge_num = len(self.edge_sequence)
        self.path_dict = path_dict
        self.punish_flag = punish_flag
        self.valid_flag = valid_flag
        # params for caculate reward
        self.base_rate=base_rate
        self.elastic_rate = elastic_rate
        self.static_rate = static_rate
        self.decrease_with_edge = decrease_with_edge
        self.make_edge_on_which_path_dict()
        if self.valid_flag:
            self.create_new_param()
    # create a new request list and set of params
    def create_new_param(self):
        self.current_request:Request = None
        self.accepted_request_heap:list[Request] = []
        self.accepted_request_action_dict:Dict[str, int]= {}
        self.remain_dict = copy.deepcopy(self.original_remain_dict)
        self.elastic_hist_dist:Dict[str, List[float]] = {}
        self.elastic_hist_list:Dict[str, List[Request]] = {}
        self.elastic_hist_sum:Dict[str, List[float]] = {}

        self.elastic_initial_action_dict:Dict[str,int] = {}
        # request in queue
        self.request_in_edge:Dict[str, List[Request]] = {}
        for edge in self.edge_sequence:
            self.request_in_edge[edge] = []

        #intiallize all request
        self.total_request_list:list[Request] = []
        self.initialize_elastic_history()
        self.initialize_all_request()
        self.total_request_list.sort(key=lambda  x: x.arrival_stamp)

    def reset(self):
        if not self.valid_flag:
            self.create_new_param()
        self.current_request = self.total_request_list.pop(0)
        return self.make_observation()

    def initialize_all_request(self):
        self.total_request_list = []
        for request_type in self.request_type_list:
            self.total_request_list += \
                self.produce_on_request_type(request_type) 
    
    def initialize_elastic_history(self):
        for request_type in self.request_type_list:
            if not request_type.isStatic:
                self.elastic_hist_dist[request_type.id] = \
                    [0.0 for _ in request_type.distribution_list]
                self.elastic_hist_sum[request_type.id] = \
                    [0.0 for _ in request_type.distribution_list]
                self.elastic_hist_list[request_type.id] = []

    def update_elastic_history(self):
        id = self.current_request.type.id
        # update accumulated sum
        if self.current_request.isScale:
            self.elastic_hist_sum[id][1] \
                += self.current_request.service_time
        else:
            self.elastic_hist_sum[id][0] \
                += self.current_request.service_time
        # update buffer list
        if len(self.elastic_hist_list[id])> \
            self.elastic_history_buffer_length:
            old_request = self.elastic_hist_list[id].pop(0)
            if old_request.isScale:
                self.elastic_hist_sum[old_request.type.id][1] -= \
                    old_request.service_time
            else:
                self.elastic_hist_sum[old_request.type.id][0] -= \
                    old_request.service_time

        self.elastic_hist_list[id].append(self.current_request)

        # update distribution
        total_service_time = sum(self.elastic_hist_sum[id])
        self.elastic_hist_dist[id] = [x/total_service_time \
            for x in self.elastic_hist_sum[id]]
        
    def produce_on_request_type(self,request_type):
        request_list = []
        # produce arrival time list
        arrival_time_list = []
        prev_sum  = 0
        while prev_sum < self.total_time :
            arrival_slot = np.random.exponential(request_type.arrival_rate)
            prev_sum += arrival_slot
            arrival_time_list.append(prev_sum)

        # produce service time list
        service_time_list = [np.random.exponential(request_type.service_rate) \
            for _ in arrival_time_list]

        for arrival_time, service_time in \
            zip(arrival_time_list, service_time_list):
            
            id = str(arrival_time)+'-'+str(request_type.id)
            request_list.append(Request(id,\
                request_type.source, request_type.sink, \
                    arrival_time, request_type.bandwidth_list[0], \
                        service_time, request_type))

            if not request_type.isStatic:
                # produce scale request
                # initial bandwidth 
                current_bandwidth = random.choice(request_type.bandwidth_list)
                prev_sum = 0
                while prev_sum < service_time:
                    if current_bandwidth > request_type.bandwidth_list[0]:
                        scale_bw = current_bandwidth- \
                            request_type.bandwidth_list[0]
                        # service time for scale request
                        scale_service_time = \
                            np.random.exponential(request_type.switch_rate_list[1])
                        if scale_service_time+prev_sum>service_time:
                            scale_service_time = service_time-prev_sum
                        request_list.append(Request(id,request_type.source, \
                            request_type.sink, \
                            arrival_time+prev_sum, scale_bw, \
                                scale_service_time, request_type, True))
                        prev_sum += scale_service_time
                        current_bandwidth = request_type.bandwidth_list[0]
                    else:
                        lower_bw_service_time = \
                            np.random.exponential(request_type.switch_rate_list[0])
                        if lower_bw_service_time+prev_sum>service_time:
                            lower_bw_service_time = service_time-prev_sum
                        prev_sum += lower_bw_service_time
                        current_bandwidth = request_type.bandwidth_list[1]
        return request_list

    def make_edge_on_which_path_dict(self):
        self.edge_on_which_path = {}
        for edge in self.edge_sequence:
            self.edge_on_which_path[edge] = {}
            for source_sink_pair in self.path_dict.keys():
                self.edge_on_which_path[edge][source_sink_pair]=[]
                paths = self.path_dict[source_sink_pair]

                for path_id, edge_list in paths.items():
                    if edge in edge_list:
                        self.edge_on_which_path[edge][source_sink_pair].append(1)
                    else:
                        self.edge_on_which_path[edge][source_sink_pair].append(0)
        
    def make_observation(self):
        # edge state
        # edge - remain bd dict to list
        remain_bd_list = []
        for edge in self.edge_sequence:
            remain_bd_list.append(self.remain_dict[edge])
            # check the edge on which path 
            source_sink_pair = self.get_source_sink_pair(self.current_request)
            remain_bd_list += self.edge_on_which_path[edge][source_sink_pair]

        edge_state_encode = np.array(remain_bd_list, dtype=np.float32)
        # current request (source one hot, sink one hot, request bd, 
        # service time, type one hot)
        cur_req_encode = self.req_2_vector(self.current_request)
        obs_part1 = np.concatenate([edge_state_encode,cur_req_encode])
        obs_part2 = []
        # request in service
        for edge in self.edge_sequence:
            request_list = self.request_in_edge[edge]
            if len(request_list)==0:
                obs_part2.append(np.zeros(\
                    (1,1,self.req_encode_size),dtype=np.float32))
            else:
                obs_part2.append(self.transform_list(request_list))
        # for edge in self.edge_sequence:
        #     request_list = self.request_in_edge[edge]
        #     if len(request_list)==0:
        #         continue
        #     else:
        #         obs_part2+=request_list
        # if len(obs_part2)==0:
        #     obs_part2 = np.zeros((1,1,self.req_encode_size),dtype=np.float32)
        # else:
        #     obs_part2 = self.transform_list(obs_part2)

        return obs_part1, obs_part2, self.current_request
        # return obs_part1

    def req_2_vector(self, req:Request):
        cur_req_encode = np.zeros(self.req_encode_size,dtype=np.float32)
        source = req.source
        sink = req.sink
        cur_req_encode[source] = 1
        cur_req_encode[self.node_num+sink]=1
        cur_req_encode[self.node_num*2]=req.bandwidth
        cur_req_encode[self.node_num*2+1]=req.service_time

        if req.type.isStatic:
            cur_req_encode[self.node_num*2+1+1+0] = 1 
        elif req.isScale:
            cur_req_encode[self.node_num*2+1+1+1] = 1
        else:
            cur_req_encode[self.node_num*2+1+1+2] = 1
        return cur_req_encode

    def transform_list(self, req_list:List[Request]):
        random.shuffle(req_list)
        req_vector_list = []
        for req in req_list:
            req_vector_list.append(self.req_2_vector(req))
        return np.expand_dims(np.vstack(req_vector_list),axis=0)

    def deactivate_all_following_elastic(self):
        # remove all following scaling request
        # make sure current request is elastic initial request
        if not self.current_request.type.isStatic and \
            not self.current_request.isScale:
            for request in self.total_request_list:
                if request.arrival_stamp > self.current_request.leave_stamp:
                    break
                if request.id == self.current_request.id:
                    # new_total_request_list.append(request)
                    request.isActivate = False

    def step(self, action):
        finished = False
        if not self.current_request.type.isStatic:
            self.update_elastic_history()
        # if self.current_request.isScale:
        #     self.update_elastic_history()

        # accpet
        reward = self.reward(action)
        if action != 0:
            if self.valid_deploy(action):
                heapq.heappush(self.accepted_request_heap, self.current_request)
                id = self.get_unique_id(self.current_request)
                self.accepted_request_action_dict[id]=action
                self.update_env(action)
            else:
                self.deactivate_all_following_elastic()
                if self.punish_flag:
                    reward *= -1.
                else:
                    reward = 0
                    finished = True
                    return self.make_observation(), reward, finished, {}
        else:
            self.deactivate_all_following_elastic()

        # finished
        if len(self.total_request_list)<=1:
            finished = True
            return self.make_observation(), reward, finished, {}

        # update current request
        request = self.total_request_list.pop(0)
        while not request.isActivate and len(self.total_request_list)>0:
            request = self.total_request_list.pop(0)
        self.current_request = request

        if len(self.total_request_list)<=1:
            finished = True
            return self.make_observation(), reward, finished, {}
            
        # pop out all leaved request
        while len(self.accepted_request_heap)>0 and \
            self.accepted_request_heap[0].leave_stamp<self.current_request.arrival_stamp:
            old_request = heapq.heappop(self.accepted_request_heap)
            id = self.get_unique_id(old_request)
            path_idx = self.accepted_request_action_dict[id]
            # free usage of each edge
            # delete corresponding request in edge 
            # find corresponding path
            affected_edges = self.path_dict\
                [self.get_source_sink_pair(old_request)][path_idx]
            for edge in affected_edges:
                self.remain_dict[edge] += old_request.bandwidth
                self.remove_request_in_service(edge, old_request)

        finished = False
        return self.make_observation(), reward, finished, {}

    def get_source_sink_pair(self, request:Request):
        return str(request.source)+'-'+str(request.sink)
    def get_unique_id(self, request:Request):
        return str(request.arrival_stamp)+\
                str(request.leave_stamp)+\
                '-'+str(request.type.id)

    def remove_request_in_service(self, edge:str, remove_request:Request):
        req_list = self.request_in_edge[edge]
        new_req_list:List[Request] = []
        for req in req_list:
            if self.get_unique_id(req) == self.get_unique_id(remove_request):
                continue
            else:
                new_req_list.append(req)
        self.request_in_edge[edge] = new_req_list

    def reward(self,action):
        
        # accept 
        if not action==0:
            # count edge num in path
            num_edge_on_path = len(self.path_dict\
                [self.get_source_sink_pair(self.current_request)][action])
            base_reward = self.current_request.bandwidth \
                * self.current_request.service_time * \
                    math.pow(self.decrease_with_edge, num_edge_on_path)
            if self.current_request.type.isStatic:
                return base_reward * self.base_rate * self.static_rate
            else:
                return base_reward * self.base_rate * self.elastic_rate
        else:
            if not self.current_request.isScale:
                return 0
            else:
                base_reward = self.current_request.bandwidth * \
                    self.current_request.service_time * \
                        self.base_rate * self.elastic_rate
                hist_dist = self.elastic_hist_dist[self.current_request.type.id]
                report_dist = self.current_request.type.distribution_list
                # KL 
                KL = kl_divergence(hist_dist,report_dist)
                print(math.exp(-KL))
                return -1.0*math.exp(-KL)*base_reward

    def valid_deploy(self, action):
        affect_edges = self.path_dict\
            [self.get_source_sink_pair(self.current_request)][action]
        for edge in affect_edges:
            if self.remain_dict[edge] < self.current_request.bandwidth:
                return False
        return True

    def update_env(self, action):
        affect_edges = self.path_dict\
            [self.get_source_sink_pair(self.current_request)][action]
        for edge in affect_edges:
            self.remain_dict[edge] -= self.current_request.bandwidth
            self.request_in_edge[edge].append(self.current_request)

    def render(self, mode='human'):
        pass
    def close (self):
        pass
