from env_evaluate_RNN import EvaluateEnvironment
from request import Request
from request_type import ElasticRequestType, StaticRequestType
import pickle
dist1 = StaticRequestType(id=0, bandwidth=2, coming_rate=1, service_rate=1)
dist2 = StaticRequestType(id=1, bandwidth=5, coming_rate=1, service_rate=1)
dist3 = ElasticRequestType(id=2, bandwidth_list=[2,8], coming_rate=1,\
    service_rate=1, distribution=[0.5, 0.5],\
    switch_mu_matrix=[[-50, 50],[50,-50]])
# dist4 = ElasticRequestType(id=2, bandwidth_list=[1,9], coming_rate=2,\
#     service_rate=2, distribution=[0.1, 0.9],\
#     switch_mu_matrix=[[-10, 10],[90,-90]])
# dist4 = ElasticRequestType(id=3, bandwidth_list=[2,4,6], coming_rate=4,\
#     service_rate=4, distribution=[0.7,0.2,0.1],\
#     switch_mu_matrix=[[-10, 10, 0],[20, -40, 20],[30,10, -40]])
# request_type_list = [dist1,dist2,dist3, dist4]
request_type_list = [dist1,dist2,dist3]

NUM_TEST_REQUEST = 300
TOTAL_BANDWIDTH = 15
env_valid_list = []
for i in range(5):
    env_valid_list.append(EvaluateEnvironment(total_bandwidth = TOTAL_BANDWIDTH,\
    request_type_list=request_type_list, \
    num_of_each_request_type_list=[NUM_TEST_REQUEST for _ in request_type_list]))
env_test_list=[]
for i in range(10):
    env_test_list.append(EvaluateEnvironment(total_bandwidth = TOTAL_BANDWIDTH,\
    request_type_list=request_type_list, \
    num_of_each_request_type_list=[NUM_TEST_REQUEST for _ in request_type_list]))

filehandler = open('15-300-3_valid_env_list.obj', 'wb')
pickle.dump(env_valid_list, filehandler)

filehandler = open('15-300-3_test_env_list.obj', 'wb')
pickle.dump(env_test_list, filehandler)

