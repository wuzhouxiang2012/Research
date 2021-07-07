from typing import List


class StaticRequestType:
    def __init__(self, id, bandwidth, coming_rate, service_rate):
        self.id = id
        self.bandwidth = bandwidth
        self.coming_rate = coming_rate
        self.service_rate = service_rate
        self.isStatic = True

class ElasticRequestType:
    def __init__(self, id, bandwidth_list, coming_rate, service_rate,\
        distribution, switch_mu_matrix):
        self.id = id
        self.bandwidth_list = bandwidth_list
        self.coming_rate = coming_rate
        self.service_rate = service_rate
        self.switch_mu_matrix = switch_mu_matrix
        self.distribution = distribution
        self.isStatic = False