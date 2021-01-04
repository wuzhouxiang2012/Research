from distribution import Distribution
class Request:
    def __init__(self, arrival_stamp:float, bandwidth:int, service_time:float, distribution:Distribution):
        self.arrival_stamp = arrival_stamp
        self.bandwidth = bandwidth
        self.service_time = service_time
        self.distribution = distribution
        self.leave_stamp = arrival_stamp+service_time
    def __lt__(self, other):
        return self.leave_stamp<=other.leave_stamp

    def __repr__(self):
        return f'arrival_stamp{self.arrival_stamp:.3f},leave_stamp{self.leave_stamp:.3f},bandwidth {self.bandwidth},service time {self.service_time:.3f}, distribution id {self.distribution.id}'