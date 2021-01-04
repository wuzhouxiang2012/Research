from typing import List


class Distribution:
    def __init__(self, id:int, vals:List[float], probs:List[float])->None:
        self.id = id
        self.isStatic = len(vals)==1
        self.vals = vals
        self.probs = probs