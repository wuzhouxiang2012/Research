from env import Environment
import pickle
from typing import Dict, List
from request_type import RequestType
from util import evaluate_reject_when_full
'''
id:int, source:int, sink:int, \
arrival_rate:float, service_rate:float, \
bandwidth_list:List[float], distribution_list:List[float]=None, \
switch_rate_list:List[float]=None
'''
# # a->b (0->1)
# a_b_static_1 = RequestType(0, 0, 1, 0.75, 0.5, [2])
# a_b_static_2 = RequestType(1, 0, 1, 1.5, 1, [8])
# a_b_elastic = RequestType(2, 0, 1, 1.5, 1, [4,9], [0.8,0.2], [0.08,0.02])

# # c->d (2,3)
# c_d_static_1 = RequestType(3, 2, 3, 1.5, 1, [1])
# c_d_static_2 = RequestType(4, 2, 3, 0.75, 0.5, [7])
# c_d_elastic = RequestType(5, 2, 3, 3, 2, [3,13], [0.9, 0.1], [0.09,0.01])

# # e->f (4,5)
# e_f_static_1 = RequestType(6, 4, 5, 0.75, 0.5, [3])
# e_f_static_2 = RequestType(7, 4, 5, 1.5, 1, [6])
# e_f_elastic = RequestType(8, 4, 5, 3, 2, [5,8], [0.7, 0.3], [0.07,0.03])

# a->b (0->1)
a_b_static_1 = RequestType(0, 0, 1, 0.75, 0.5, [6])
a_b_static_2 = RequestType(1, 0, 1, 1.5, 1, [9])
a_b_elastic = RequestType(2, 0, 1, 1.5, 1, [4,9], [0.8,0.2], [0.08,0.02])

# c->d (2,3)
c_d_static_1 = RequestType(3, 2, 3, 1.5, 1, [5])
c_d_static_2 = RequestType(4, 2, 3, 0.75, 0.5, [9])
c_d_elastic = RequestType(5, 2, 3, 3, 2, [3,13], [0.9, 0.1], [0.09,0.01])

# e->f (4,5)
e_f_static_1 = RequestType(6, 4, 5, 0.75, 0.5, [4])
e_f_static_2 = RequestType(7, 4, 5, 1.5, 1, [12])
e_f_elastic = RequestType(8, 4, 5, 3, 2, [3,13], [0.7, 0.3], [0.07,0.03])
#Total Time 6
total_time = 600

'''
remain_dict:Dict[str,float],\
node_num:int,\
path_dict:Dict[str,Dict[int,List[str]]],\
request_type_list:List[RequestType], \
total_time:float,\
base_rate:float=1, elastic_rate:float=1.1, \
static_rate:float=0.9,\
elastic_history_buffer_length:int=100,\
decrease_with_edge:float = 0.9
'''

remain_dict = {'0-1':10, '0-2':10, '1-3':10, \
    '2-3':20, '2-4':10, '3-5':10, '4-5':10}
edge_sequence = ['0-1', '0-2', '1-3', '2-3', '2-4', '3-5', '4-5']
node_num = 6
path_dict:Dict[str,Dict[int,List[str]]] = \
    {'0-1':{1:['0-1'], \
            2:['0-2','2-3','1-3'],\
            3:['0-2','2-4','4-5','3-5','1-3']},\
     '2-3':{1:['2-3'],\
            2:['0-2','0-1','1-3'], \
            3:['2-4','4-5','3-5']}, \
     '4-5':{1:['4-5'],\
            2:['2-4','2-3','3-5'],\
            3:['2-4','0-2','0-1','1-3','3-5']}}

request_type_list = \
    [a_b_static_1,a_b_static_2,a_b_elastic,\
        c_d_static_1, c_d_static_2, c_d_elastic, \
            e_f_static_1, e_f_static_2, e_f_elastic]



def produce_test_env_list(NUM_ENV, path):
    env_list = []
    for i in range(NUM_ENV):
        environment = Environment(remain_dict=remain_dict, \
            edge_sequence=edge_sequence,\
            node_num=node_num,\
            path_dict=path_dict, request_type_list=request_type_list,\
            total_time=total_time,\
            punish_flag=True, valid_flag=True)
        env_list.append(environment)
    filehandler = open(path,"wb")
    pickle.dump(env_list, filehandler)
    filehandler.close()

def produce_env(total_time=600, punish_flag=True, valid_flag=False):
    environment = Environment(remain_dict=remain_dict, \
                edge_sequence=edge_sequence,\
                node_num=node_num,\
                path_dict=path_dict, \
                request_type_list=request_type_list,\
                total_time=total_time,\
                punish_flag=punish_flag, valid_flag=valid_flag 
                )
    return environment 

if __name__ == '__main__':
    from util import reject_when_full
    # environment = produce_env(total_time=600, punish_flag=True, valid_flag=False)
    # print(reject_when_full(environment))
    produce_test_env_list(10, 'env_list_set2')
    print(evaluate_reject_when_full('env_list_set2'))
    
