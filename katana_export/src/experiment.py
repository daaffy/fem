from aggs import *
import time
import pickle
# eps,n,eval_points,agg_sd,time_sd,kappa_history

# unsure of what to call the object
class experiment():
    def __init__(self,
                eps,
                eval_points,
                n = 0,
                t = 0,
                agg_list = aggs(3),
                kappa_history = []):             
        self.eps = eps
        self.eval_points = eval_points
        self.n = n
        self.t = t
        self.agg_list = agg_list
        self.kappa_history = kappa_history

    def update(self,kappa,sol_list):
        self.kappa_history.append(kappa)
        self.n += 1
        self.agg_list.update(sol_list)

    def begin(self):
        self.t_0 = time.time()

    def end(self):
        self.t = time.time() - self.t_0
        self.agg_list.finalise()

    # below is probably not necessary, as one can just 'pickle' the experiment object from the 'outside'
    def save(self,dir_str):
        with open(dir_str,'wb') as f:
            pickle.dump([   
                self.eps,
                self.n,
                self.eval_points,
                self.agg_list,
                self.t,
                self.kappa_history
                ],f)
            f.close()

    @classmethod
    def load(cls,dir_str):
        with open(dir_str, 'rb') as f:
            eps,n,eval_points,agg_list,t,kappa_history = pickle.load(f)
            f.close()
        return cls(eps,eval_points,n,t,agg_list,kappa_history)

    # combine stats from a list of experiments (same epsilon)
    @classmethod
    def combine(cls,e_list):
        n_e = len(e_list)
        e0 = e_list[0]
        eps = e0.eps
        eval_points = e0.eval_points
        n = e0.n
        t = e0.t
        agg_list = e0.agg_list
        kappa_history = e0.kappa_history
        for i in range(1,n_e):
            assert eps == e_list[i].eps # check all epsilons are equal
            # (!) check eval_points are equal
            n += e_list[i].n
            t += e_list[i].t # should we add time or append to list?
            agg_list.combine(e_list[i].agg_list)
            kappa_history.append(e_list[i].kappa_history)
        return cls(eps,eval_points,n,t,agg_list,kappa_history)
        