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

    def combine(self,new_aggs):
        assert new_aggs.agg_size == self.agg_size # must be the same length to combine
        for i in range(self.agg_size):
            self[i].combine(new_aggs[i])
        return

    def finalise(self):
        for i in range(self.agg_size):
            self[i].finalise()

class agg():
    def __init__(self):
        self.curr = (0,0,0) # initialize agg tuple
        self.statistics = (0,0,0)

    def update(self,new_data):
        self.curr = self.__update(self.curr,new_data)

    # i think i'd prefer to make this a class method
    def combine(self,new_agg):
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        n_a = self.curr[0]; n_b = new_agg.curr[0]
        mean_a = self.curr[1]; mean_b = new_agg.curr[1]
        M_a = self.curr[2]; M_b = new_agg.curr[2]

        n_ab = n_a + n_b
        delta = mean_b - mean_a
        mean_ab = mean_a + delta*n_b/(n_a+n_b)
        M_ab = M_a + M_b + delta**2*n_a*n_b/(n_a+n_b)

        self.curr = (n_ab,mean_ab,M_ab)

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