from typing import List
class RequestType:
    def __init__(self, id:int, source:int, sink:int, \
        arrival_rate:float, service_rate:float, \
        bandwidth_list:List[float], distribution_list:List[float]=None, \
        switch_rate_list:List[float]=None) -> None:

        self.id = id
        self.source = source
        self.sink = sink
        self.bandwidth_list = bandwidth_list
        self.distribution_list = distribution_list
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.switch_rate_list = switch_rate_list
        self.isStatic:bool = len(self.bandwidth_list)==1
        