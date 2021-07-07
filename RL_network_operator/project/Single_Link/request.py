from request_type import ElasticRequestType, StaticRequestType
class Request:
    def __init__(self, id, arrival_stamp, bandwidth, service_time, type, isScale=False):
        self.id = id
        self.arrival_stamp = arrival_stamp
        self.bandwidth = bandwidth
        self.service_time = service_time
        self.type = type
        self.leave_stamp = arrival_stamp+service_time
        self.isScale = isScale
    def __lt__(self, other):
        return self.leave_stamp<=other.leave_stamp

    def __repr__(self):
        return f'id:{self.id}, arrival_stamp{self.arrival_stamp:.3f},leave_stamp{self.leave_stamp:.3f},bandwidth {self.bandwidth},service time {self.service_time:.3f}, distribution id {self.type.id}\n'