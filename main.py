#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import config
import generators
import methods
import measure
from common import cost, normalize_cols, hellinger, print_head, get_permute
import visualize
import data
import prepare
#from numpy import linalg

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as ml
from datetime import datetime, timedelta
from time import time
from copy import deepcopy


def gen_real(cfg=config.default_config()):
    """Generate matrices with real values for model experiment.
       - Return:
       F
       Phi_r
       Theta_r
       - Used params:
       N
       T_0
       M
       gen_phi
       phi_sparsity
       gen_theta
       theta_sparsity
    """
    N = cfg['N']
    T_0 = cfg['T_0']
    M = cfg['M']

    gen_phi = getattr(generators, cfg['gen_phi'])
    cfg['rows'] = N
    cfg['cols'] = T_0
    cfg['sparsity'] = cfg['phi_sparsity']
    Phi_r = gen_phi(cfg)

    gen_theta = getattr(generators, cfg['gen_theta'])
    cfg['rows'] = T_0
    cfg['cols'] = M
    cfg['sparsity'] = cfg['theta_sparsity']
    Theta_r = gen_theta(cfg)

    F = np.dot(Phi_r, Theta_r)
    return (F, Phi_r, Theta_r)


def gen_init(cfg=config.default_config()):
    """Generate real valued initialization matrices.
       - Return:
       Phi
       Theta
       - Used params:
       N
       T
       M
       gen_phi
       phi_sparsity
       gen_theta
       theta_sparsity
    """
    N = cfg['N']
    T = cfg['T']
    M = cfg['M']

    gen_phi = getattr(generators, cfg['phi_init'])
    cfg['rows'] = N
    cfg['cols'] = T
    cfg['sparsity'] = cfg['phi_sparsity']
    Phi = gen_phi(cfg)

    gen_theta = getattr(generators, cfg['theta_init'])
    cfg['rows'] = T
    cfg['cols'] = M
    cfg['sparsity'] = cfg['theta_sparsity']
    Theta = gen_theta(cfg)

    return (Phi, Theta)


def run(F, Phi, Theta, Phi_r=None, Theta_r=None, cfg=config.default_config()):
    """Em-algo method.
       - Return:
       val
       hdist
       it
       Phi
       Theta
       status
       - Used params:

    """
    T = Theta.shape[0]
    eps = cfg['eps']
    schedule = cfg['schedule'].split(',')
    meas = cfg['measure'].split(',')
    val = np.zeros((cfg['max_iter']+2, len(meas)))
    hdist = np.zeros((cfg['max_iter']+2, 1))
    
    for i, fun_name in enumerate(meas):
        fun = getattr(measure, fun_name)
        val[0, i] = fun(F, np.dot(Phi, Theta))
    
    if cfg['compare_real']:
        #m = Munkres()
        idx = get_permute(Phi_r, Theta_r, Phi, Theta, cfg['munkres'])
        hdist[0] = hellinger(Phi[:, idx[:, 1]], Phi_r[:, idx[:, 0]]) / T
    if cfg['print_lvl'] > 1:
        print('Initial loss:', val[0])
    status = 0
    methods_num = len(schedule)
    it = -1
    for it in range(cfg['max_iter']):
        if cfg['print_lvl'] > 1:
            print('Iteration', it+1)
        Phi_old = deepcopy(Phi)
        Theta_old = deepcopy(Theta)
        method_name = schedule[it % methods_num]
        if cfg['print_lvl'] > 1:
            print('Method:', method_name)
        method = getattr(methods, method_name)
        (Phi, Theta) = method(F, Phi, Theta, method_name, cfg)
        if (it+1) % cfg['normalize_iter'] == 0:
            Phi = normalize_cols(Phi)
            Theta = normalize_cols(Theta)
        for j, fun_name in enumerate(meas):
            fun = getattr(measure, fun_name)
            val[it+1, j] = fun(F, np.dot(Phi, Theta))
        
        if cfg['compare_real']:
            idx = get_permute(Phi_r, Theta_r, Phi, Theta, cfg['munkres'])
            hdist[it+1] = hellinger(Phi[:, idx[:, 1]], Phi_r[:, idx[:, 0]]) / T
        
        if cfg['print_lvl'] > 1:
            print(val[it+1])
        if all(val[it, :] < eps):
            if cfg['print_lvl'] > 1:
                print('By cost.')
            status = 1
            break
        if abs(Phi_old - Phi).max() < eps and abs(Theta_old - Theta).max() < eps:
            if cfg['print_lvl'] > 1:
                print('By argument.')
            status = 2
            break
        #del W_old
        #del H_old
    if cfg['print_lvl'] > 1:
        print('Final:')
    Phi = normalize_cols(Phi)
    Theta = normalize_cols(Theta)
    for j, fun_name in enumerate(meas):
        fun = getattr(measure, fun_name)
        val[it+2:, j] = fun(F, np.dot(Phi, Theta))
    
    if cfg['compare_real']:
        idx = get_permute(Phi_r, Theta_r, Phi, Theta, cfg['munkres'])
        hdist[it+2:] = hellinger(Phi[:, idx[:, 1]], Phi_r[:, idx[:, 0]]) / T
    return (val, hdist, it, Phi, Theta, status)

def load_dataset(cfg=config.default_config()):
    """Load or generate dataset.
       - Return:
       F
       vocab
       N
       M
       Phi_r
       Theta_r 
       - Used params:
       load_data
       data_name?
    """
    if cfg['load_data'] == 'uci' or cfg['load_data'] == 1:
        print("uci")
        F, vocab = data.load_uci(cfg['data_name'], cfg)
        N, M = F.shape
        F = normalize_cols(F)
        cfg['N'], cfg['M'] = F.shape
        print('Dimensions of F:', N, M)
        print('Checking assumption on F:', np.sum(F, axis=0).max())
        return F, vocab, N, M, None, None
    else:
        F, Phi_r, Theta_r = gen_real(cfg)
        print('Checking assumption on F:', np.sum(F, axis=0).max())
        return F, None, F.shape[0], F.shape[1], Phi_r, Theta_r

def initialize_matrices(i, F, cfg=config.default_config()):
    """Initialize matrices Phi Theta.
       - Return:
       Phi
       Theta
       - Used params:
       prepare_method
    """
    if (int(cfg['prepare_method'].split(',')[i]) == 1):
        print("Arora")
        eps = cfg['eps']
        Phi = prepare.anchor_words(F, 'L2', cfg)
        print('Solving for Theta')
        Theta = np.linalg.solve(np.dot(Phi.T, Phi) + np.eye(Phi.shape[1]) * eps, np.dot(Phi.T, F))
        Theta[Theta < eps] = 0
        Theta = normalize_cols(Theta)
        return Phi, Theta
    elif (int(cfg['prepare_method'].split(',')[i]) == 2):
        print("Random")
        return gen_init(cfg)

def calculate_stats(series):
    series_mean = np.mean(series, axis=0)
    series_var = np.var(series, axis=0)
    series_min = series[np.argmin(series[:,-1]),:]
    series_max = series[np.argmax(series[:,-1]),:]
    return series_mean, np.sqrt(series_var), series_min, series_max

def main(config_file='config.txt', results_file='results.txt', cfg=None):
    """Main function which runs experiments.
       - Return:
       res
       - Used params:
       N
       T
       M
       eps
       seed?
       run_info
       measure
       compare_methods
       schedule
       compare_real
       save_topics
    """
    if cfg == None:
        cfg = config.load(config_file)
    if cfg['seed'] >= 0:
        np.random.seed(cfg['seed'])
    else:
        np.random.seed(None)

    eps = cfg['eps']
    N = cfg['N']
    T = cfg['T']
    M = cfg['M']
    vocab = None
    Phi_r = None
    Theta_r = None

    if cfg['run_info'] == 'results' or cfg['run_info'] == 1:
        cfg['print_lvl'] = 1
    elif cfg['run_info'] == 'run' or cfg['run_info'] == 2:
        cfg['print_lvl'] = 2
    else:
        cfg['print_lvl'] = 0
    if cfg['print_lvl'] > 0:
        print('Generating matrices...')

    #loading dataset or generating new
    F, vocab, N, M, Phi_r, Theta_r = load_dataset(cfg)
    
    #loading expirement information
    total_runs = sum([int(x) for x in cfg['runs'].split(",")])
    results = [0] * total_runs #res==results, different quality measures arrays
    finals = [0] * total_runs
    hdist_runs = [0] * total_runs #hellinger distances arrays
    exp_time = [0] * total_runs #time for run EM-algo

    #measures to implement
    meas = cfg['measure'].split(',')
    meas_name = [''] * len(meas)
    print('Used measures:')
    for i, f_name in enumerate(meas):
        f = getattr(measure, f_name + '_name')
        meas_name[i] = f()
        print(f_name)

    #plsa or others
    if cfg['compare_methods']:
        methods = cfg['schedule'].split(',')
        nmethods = len(methods)

    #initialization
    current_exp = 0
    for it, expirement_runs in enumerate([int(x) for x in cfg['runs'].split(",")]):
        for r in range(expirement_runs):
            if cfg['print_lvl'] > 0:
                print('Run', r+1,'/',expirement_runs, 'of expirement', it+1)
                print('  Starting...')
            labels = None
            start_time = time()
            Phi, Theta = None, None

            print('Preparing data...')
            Phi, Theta = initialize_matrices(it, F, cfg)
            end_time = time() - start_time
            print('Preparing took time:', timedelta(seconds=end_time))

            #choose one method for compare (plsa)
            if cfg['compare_methods'] > 0:
                cfg['schedule'] = methods[0]

            #calculate usual EM-alg
            start = time()
            (val, hdist, i, Phi, Theta, status) = run(F, Phi, Theta, Phi_r, Theta_r, cfg)
            stop = time()
            print('Run time:', timedelta(seconds=stop - start))

            #write results
            exp_time[current_exp] = stop - start
            results[current_exp] = val
            hdist_runs[current_exp] = hdist
            if cfg['print_lvl'] > 0:
                print('  Result:', val[-1, :])
            for i, fun_name in enumerate(cfg['finals'].split(',')):
                fun = getattr(measure, fun_name)
                name, val = fun(Phi, Theta)
                print(name, ':', val)

            #save results for different runs
            if cfg['save_topics']:
                visualize.save_topics(Phi, os.path.join(cfg['result_dir'], cfg['experiment'] + '_'+str(current_exp)+'topics.txt'), vocab)
            if cfg['compare_real']:
                visualize.show_matrices_recovered(Phi_r, Theta_r, Phi, Theta, cfg, permute=True)
            current_exp += 1

    #save results section
    if cfg['experiment'] == '':
        exp_name = 'test'
    else:
        exp_name = cfg['experiment']

    #TODO:check
    if cfg['show_results']:
        if not os.path.exists(cfg['result_dir']):
            os.makedirs(cfg['result_dir'])
        np.savetxt(os.path.join(cfg['result_dir'], cfg['experiment'] + '_Phi.csv'), Phi)
        
        for i, fun_name in enumerate(cfg['measure'].split(',')):
            val = np.array([r[:, i] for r in results])
            fun = getattr(measure, fun_name + '_name')
            visualize.plot_measure(val.T, fun())
            filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_'+fun_name+'.pdf')
            plt.savefig(filename, format='pdf')
        if cfg['compare_real']:
            print('Hellinger res:', hdist_runs[0][-1,0])
            visualize.plot_measure(np.array([r[:, 0] for r in hdist_runs]).T, measure.hellinger_name())

    return results

if __name__ != '__main__':
    main()
    plt.show()

if __name__ == '__main__':
    print("Loading config...")
    cfg = config.load()
    print("Config is loaded")
    if not os.path.exists(cfg['result_dir']):
        os.makedirs(cfg['result_dir'])

    print("Calculations:")
    results = main(cfg=cfg)

    print("Plot graphs")
    colors = ['r', 'b', 'g', 'm']
    markers = ['o', '^', 'd', (5,1)]
    for i, fun_name in enumerate(cfg['measure'].split(',')):
        plt.figure()
        val = np.array([r[:, i] for r in results])
        fun = getattr(measure, fun_name + '_name')
        plt.ylabel(fun(), fontsize=13)

        index_exp_series = 0
        for it, expirement_runs in enumerate([int(x) for x in cfg['runs'].split(",")]):
            series_stats = calculate_stats(val[index_exp_series:index_exp_series+expirement_runs, 0:])
            plt.plot(series_stats[0], linewidth=2, c=colors[it % len(colors)], label = str(it+1))
            plt.fill_between(range(len(series_stats[0])), series_stats[0] + series_stats[1], series_stats[0] - series_stats[1], alpha = 0.1, facecolor=colors[it % len(colors)])
            plt.plot(series_stats[2], linewidth=0.5, c=colors[it % len(colors)])
            plt.plot(series_stats[3], linewidth=0.5, c=colors[it % len(colors)])
            index_exp_series += expirement_runs

        plt.legend()
        plt.draw()

        filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_'+fun_name+'.pdf')
        plt.savefig(filename, format='pdf')
    plt.show()
