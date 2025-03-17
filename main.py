import numpy as np
import observation as Obs
import argparse
import bayesian_optimization as bo
import genetic_algorithm as ga
import yaml
import json
import time
import sys
import katpoint
from tabulate import tabulate
from datetime import datetime
import ast


def fit_function(obs, x, args, configs, optim = False):
    score = obs.simulate_schedule(start = args.start, timespan = configs['plan'][args.plan], method = args.selection, sb_value = [x[0], x[1], x[2], x[3]], optim = optim)
    return score

def params_dictionary(params):
    return {'coeff_a': params[0], 'coeff_b1': params[1], 'coeff_b2': params[2], 'coeff_none': params[3]}

def printing_output(score_baseline, score_optimized, algo):
    baseline = ["baseline score", score_baseline[0], score_baseline[1], score_baseline[2], score_baseline[3]]
    optimized = ["optimized score (%s)" % algo, score_optimized[0], score_optimized[1], score_optimized[2], score_optimized[3]]
    data = [baseline, optimized]
    table = tabulate(
                        data, 
                        headers=["Score name", "# idle", "# A-rank", "# B1-rank", "# B2-rank"], 
                        tablefmt="grid"
                    )
    print(table)

def assert_arguments(args):
    assert args.plan == 'long' or args.plan == 'short', 'plan has only two values: long or short [as string]'
    assert args.algo == 'bo' or args.algo == 'ga', 'algo has only two values: bo or ga [as string]'
    assert args.optim == 'True' or args.optim == 'False', 'optim has only two values: True or False [as string]'
    assert args.save == 'True' or args.save == 'False', 'save has only two values: True or False [as string]'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scheduling Optimization')
    parser.add_argument('--selection', type = str, default = 'greedy', help = 'type of selection')
    parser.add_argument('--start', type = str, default = '2024-10-07 06:00:00', help = 'specify the start time')
    parser.add_argument('--plan', type = str, default = 'long', help = 'specify whether it is a long/short term plan')
    parser.add_argument('--algo', type = str, default = 'bo', help = 'specify the optimization algorithm to be used')
    parser.add_argument('--optim', type = str, default = 'True', help = 'specify whether to optimize or test existing params')
    parser.add_argument('--save', type = str, default = 'False', help = 'specify whether to save the optimized params')
    parser.add_argument('--sched', type = str, default = 'False', help = 'specify whether to save the resulting scheduling plan')
    args = parser.parse_args()
    assert_arguments(args)
    # print(ast.literal_eval(args.sched))
    print('*' * 20)
    print('%s-term planning' % args.plan)
    print('*' * 20)
    configs = yaml.safe_load(open('config.yml'))
    obs = Obs.Observation(configs, args)
    print('estimating the baseline ...')
    baseline = fit_function(obs, [4, 3, 2, 1], args, configs, optim = False)
    if args.optim == 'True':
        if args.algo == 'bo':
            start_time_it = time.time()
            print('building the bayesian model for optimization ...')
            bayesopt = bo.BayesOpt(configs['bayesopt'], obs, args.start, configs['plan'][args.plan], args.selection, args.plan)
            print('optimization begins ...')
            bayesopt.optimize()
            params = bayesopt.build_opt.x_opt
            print('optimization takes about %4.2f min' % ((time.time() - start_time_it) / 60))
            score = fit_function(obs, params, args, configs, optim = False)
            printing_output(baseline, score[:-1], args.algo)
        elif args.algo == 'ga':
            start_time_it = time.time()
            print('building the genetic algorithm model for optimization ...')
            geneticalgo = ga.GeneticAlgo(configs['geneticalgo'], obs, args.start, configs['plan'][args.plan], args.selection, args.plan)
            print('optimization begins ...')
            geneticalgo.optimize()
            print('optimization takes about %4.2f min' % ((time.time() - start_time_it) / 60))
            params = geneticalgo.model.best_variable
            score = fit_function(obs, params, args, configs, optim = False)
            printing_output(baseline, score[:-1], args.algo)
            # print(score[4])
        if args.save == 'True':
            dict_params = params_dictionary(params)
            with open('params/%s_%s_params.json' % (args.algo, args.plan), 'w') as fp:
                json.dump(dict_params, fp, sort_keys=True, indent=4)
    else:
        print('testing existing parameters ...')
        with open('params/%s_%s_params.json' % (args.algo, args.plan)) as json_file:
            params_dict = json.load(json_file)
        params = [params_dict['coeff_a'], params_dict['coeff_b1'], params_dict['coeff_b2'], params_dict['coeff_none']]
        score = fit_function(obs, params, args, configs, optim = False)
        printing_output(baseline, score[:-1], args.algo)
    if args.sched == 'True':
        score[4].to_csv('schedule/%s_%s_%s.csv' % (str(datetime.now()).split('.')[0].replace(" ", "_"), args.algo, args.plan))
