from request_type import RequestType
class Request:
    def __init__(self, id:str,source:int, sink:int, \
        arrival_stamp:float, bandwidth:float, \
        service_time:float, type:RequestType, \
        isScale:bool=False, isActivate:bool=True):

        self.id = id
        self.source = source
        self.sink = sink
        self.arrival_stamp = arrival_stamp
        self.bandwidth = bandwidth
        self.service_time = service_time
        self.type = type
        self.leave_stamp = arrival_stamp+service_time
        self.isScale = isScale
        self.isActivate = isActivate

    def __lt__(self, other):
        return self.leave_stamp<=other.leave_stamp

    def __repr__(self):
        return f'id:{self.id} \
            arrival_stamp:{self.arrival_stamp:.3f}  \
            leave_stamp:{self.leave_stamp:.3f} \n \
            bandwidth:{self.bandwidth} \
            service time:{self.service_time:.3f} \n \
            type id:{self.type.id}, isScale:{self.isScale}\n'