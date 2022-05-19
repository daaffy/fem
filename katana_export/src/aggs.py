# agg object

class aggs(list):
    def __init__(self,agg_size):
        self.agg_size = agg_size
        for i in range(self.agg_size):
            self.append(agg())

    def update(self,new_data):
        assert len(new_data) == self.agg_size
        for i in range(self.agg_size):
            self[i].update(new_data[i])

    def finalise(self):
        for i in range(self.agg_size):
            self[i].finalise()

class agg(list):
    def __init__(self):
        self.curr = (0,0,0) # initialize agg tuple

    def update(self,new_data):
        self.curr = self.__update(self.curr,new_data)

    def finalise(self):
        self.statistics = self.__finalise(self.curr)
    
    # from wiki:
    def __update(self,existingAggregate, newValue):
        (count, mean, M2) = existingAggregate
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2
        return (count, mean, M2)   

    def __finalise(self,existingAggregate):
        (count, mean, M2) = existingAggregate
        if count < 2:
            return float("nan")
        else:
            (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
            return (mean, variance, sampleVariance) # note: returns np array objects if they are fed in as input