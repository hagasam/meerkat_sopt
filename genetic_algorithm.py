import numpy as np
import random
# np.random.seed(40) 
from geneticalgorithm import geneticalgorithm as ga
import sys
import matplotlib
matplotlib.use('Agg')


class GeneticAlgo:
    def __init__(self, configs, Obs, starttime, timespan, selection, plan):
        super(GeneticAlgo, self).__init__()
        self.obs = Obs
        self.start_time = starttime
        self.timespan = timespan
        self.selection = selection
        self.plan = plan
        self.num_dimension = 4
        self.configs = configs
        self.varbound = np.array([self.configs['varabound']] * self.num_dimension)
        self.algorithm_param = {'max_num_iteration': self.configs['params']['max_num_iteration'], 
                   'population_size': self.configs['params']['population_size'],
                   'mutation_probability': self.configs['params']['mutation_probability'],
                   'elit_ratio': self.configs['params']['elit_ratio'],
                   'crossover_probability': self.configs['params']['crossover_probability'],
                   'parents_portion': self.configs['params']['parents_portion'],
                   'crossover_type': self.configs['params']['crossover_type'],
                   'max_iteration_without_improv': None}
        self.model = ga(function = self.cost_function,
                        dimension = self.num_dimension,
                        variable_type = 'real',
                        variable_boundaries = self.varbound,
                        algorithm_parameters = self.algorithm_param,
                        function_timeout = 60)

    def cost_function(self, x):
        score = self.obs.simulate_schedule(start = self.start_time, timespan = self.timespan,
                                          method = self.selection,
                                          sb_value = x, plan = self.plan)
        return score
    
    def optimize(self):
        self.model.run()