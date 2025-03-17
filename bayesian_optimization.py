import numpy as np
import random
np.random.seed(40) 
from GPyOpt.methods import BayesianOptimization
import sys


class BayesOpt:
    def __init__(self, configs, Obs, starttime, timespan, selection, plan):
        super(BayesOpt, self).__init__()
        self.obs = Obs
        self.start_time = starttime
        self.timespan = timespan
        self.selection = selection
        self.plan = plan
        self.configs = configs
        self.domain = [{'name': 'coeffA', 'type': 'continuous', 'domain': tuple(self.configs['coeffA'])},
                        {'name': 'coeffB1', 'type': 'continuous', 'domain': tuple(self.configs['coeffB1'])},
                        {'name': 'coeffB2', 'type': 'continuous', 'domain': tuple(self.configs['coeffB2'])},
                        {'name': 'coeffNone', 'type': 'continuous', 'domain': tuple(self.configs['coeffNone'])}]
        self.build_opt = BayesianOptimization(f = self.cost_function, domain = self.domain, 
                                                acquisition_type = self.configs['acquisition_type'],
                                                acquisition_weight = self.configs['acquisition_weight'], 
                                                maximize = self.configs['maximize'])
    
    def cost_function(self, x):
        x = np.atleast_2d(x)
        fs = np.zeros((x.shape[0],1))
        for i in range(x.shape[0]):
            score = self.obs.simulate_schedule(start = self.start_time, timespan = self.timespan, 
                                                method = self.selection, 
                                                sb_value = [x[i,0], x[i,1], x[i,2], x[i,3]], plan = self.plan)
            fs[i] = score
        return fs
    
    def optimize(self):
        self.build_opt.run_optimization(max_iter=self.configs['max_iter'])
