#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import print_function

import config
import generators
import methods
import measure
from common import cost, hellinger, normalize_cols, print_head, get_permute
import visualize
import data
import prepare
import pickle
#from numpy import linalg

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as ml
from datetime import datetime, timedelta
from time import time
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfTransformer
from shutil import copyfile
import main
from main import calculate_stats

'''def experiment_old():
    cfg = config.load('config.txt')
    doubles = ['als', 'mult'];
    prim = 'plsa'
    once = ['als', 'mult']
    Ts = [15]
    Tn = ['Te']
    
    norms = [1, 2]
    suff = ''
    all_res = []
    #doubles = []
    # double algorithms
    for alg in doubles:
        for norm in norms:
            for (i, T) in enumerate(Ts):
                if alg == 'als' and T > 15:
                    continue
                cfg['experiment'] = alg + '_' + prim + '_' + Tn[i] + '_' + str(norm) + suff
                cfg['T'] = T
                cfg['schedule'] = alg + ',' + prim
                cfg['normalize_iter'] = norm
                print('\n', cfg['experiment'], '\n')
                res = main.main(cfg=cfg)
    
    # one algorithm
    for alg in once:
        for (i, T) in enumerate(Ts):
            if alg == 'als' and T > 15:
                continue
            cfg['experiment'] = alg + '_' + Tn[i] + '_1' + suff
            cfg['T'] = T
            cfg['schedule'] = alg
            cfg['normalize_iter'] = 1
            print('\n', cfg['experiment'], '\n')
            res = main.main(cfg=cfg)


def experiment():
    cfg = config.load('config.txt')
    data = ['kos']
    #data = ['nips']
    methods = ['als', 'mult', 'my_als', 'plsa']
    Ts = [5, 10, 15, 20]
    prepares = [0, 1, 2, 3]
    cfg['max_iter'] = 20
    for d in data:
        for alg in methods:
            for T in Ts:
                for pr in prepares:
                    cfg['experiment'] = d+'_'+alg+'_20_T'+str(T)+'_p'+str(pr)
                    cfg['data_name'] = d
                    cfg['T'] = T
                    cfg['schedule'] = alg
                    cfg['prepare_method'] = pr
                    print('============\n', cfg['experiment'], '\n============')
                    res = main.main(cfg=cfg)
                    plt.close('all')'''

def plot_results(results, finals, cfg):
    print("Plot graphs")
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']
    markers = ['o', '^', 'd', (5,1)]
    labels = ["Arora", "Random-rare", "Random-uniform", "Clust-words", "SVD", "Clust-tfIdf"]

    with open(os.path.join(cfg['result_dir'], cfg['experiment']+'_finals.txt'),"w") as f:
        for i, fun_name in enumerate(cfg['finals'].split(',')):
            fun = getattr(measure, fun_name)
            name, val = fun(np.array([[1]]), np.array([[1]]), np.array([[1]]))
            index_exp_series = 0
            for it, expirement_runs in enumerate([int(x) for x in cfg['runs'].split(",")]):
                series_mean = np.mean(finals[i][index_exp_series:index_exp_series+expirement_runs])
                series_max = np.max(finals[i][index_exp_series:index_exp_series+expirement_runs])
                series_min = np.min(finals[i][index_exp_series:index_exp_series+expirement_runs])
                f.write(str(it+1)+" "+str(name)+" "+str(series_mean)+" "+str(series_max)+" "+str(series_min)+"\n")
                index_exp_series += expirement_runs
            f.write('\n')

    copyfile("config.txt", os.path.join(cfg['result_dir'], cfg['experiment']+'_config.txt'))

    for i, fun_name in enumerate(cfg['measure'].split(',')):
        plt.figure()
        val = np.array([r[:, i] for r in results])
        fun = getattr(measure, fun_name + '_name')
        plt.ylabel(fun(), fontsize=18)
        plt.title("F", fontsize=18)
        plt.xlabel(u"Номер итерации", fontsize=18)
        #plt.grid(True)
        index_exp_series = 0
        for it, expirement_runs in enumerate([int(x) for x in cfg['runs'].split(",")]):
            series_stats = calculate_stats(val[index_exp_series:index_exp_series+expirement_runs, 0:], cfg['begin_graph_iter'])
            plt.plot(range(cfg['begin_graph_iter'], cfg['begin_graph_iter'] + len(series_stats[0])), series_stats[0], linewidth=2, c=colors[it % len(colors)], label = labels[it])
            plt.fill_between(range(cfg['begin_graph_iter'], cfg['begin_graph_iter'] + len(series_stats[0])), series_stats[2], series_stats[3], alpha = 0.1, facecolor=colors[it % len(colors)])
            '''plt.fill_between(range(len(series_stats[0])), series_stats[0] + series_stats[1], series_stats[0] - series_stats[1], alpha = 0.1, facecolor=colors[it % len(colors)])
            plt.plot(series_stats[2], linewidth=0.5, c=colors[it % len(colors)])
            plt.plot(series_stats[3], linewidth=0.5, c=colors[it % len(colors)])'''
            index_exp_series += expirement_runs

        plt.legend()
        plt.draw()

        filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_'+fun_name+'.pdf')
        plt.savefig(filename, format='pdf')
        fun = getattr(measure, fun_name + '_name_eng')
        plt.ylabel(fun(), fontsize=18)
        plt.title("F", fontsize=18)
        plt.xlabel("Iteration", fontsize=18)
        plt.legend()
        plt.draw()

        filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_'+fun_name+'_eng.pdf')
        plt.savefig(filename, format='pdf')
        if fun_name == "perplexity" or fun_name == "frobenius":
            plt.figure()
            val = np.array([r[:, i] for r in results])
            fun = getattr(measure, fun_name + '_name')
            plt.ylabel(fun(), fontsize=18)
            plt.title("F", fontsize=18)
            plt.xlabel(u"Номер итерации", fontsize=18)
            #plt.grid(True)
            index_exp_series = 0
            for it, expirement_runs in enumerate([int(x) for x in cfg['runs'].split(",")]):
                series_stats = calculate_stats(val[index_exp_series:index_exp_series+expirement_runs, 0:], 1)
                plt.plot(range(1, 1 + len(series_stats[0])), series_stats[0], linewidth=2, c=colors[it % len(colors)], label = labels[it])
                plt.fill_between(range(1, 1 + len(series_stats[0])), series_stats[2], series_stats[3], alpha = 0.1, facecolor=colors[it % len(colors)])
                '''plt.fill_between(range(len(series_stats[0])), series_stats[0] + series_stats[1], series_stats[0] - series_stats[1], alpha = 0.1, facecolor=colors[it % len(colors)])
                plt.plot(series_stats[2], linewidth=0.5, c=colors[it % len(colors)])
                plt.plot(series_stats[3], linewidth=0.5, c=colors[it % len(colors)])'''
                index_exp_series += expirement_runs

            plt.legend()
            plt.draw()

            filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_'+fun_name+'_addit.pdf')
            plt.savefig(filename, format='pdf')
            fun = getattr(measure, fun_name + '_name_eng')
            plt.ylabel(fun(), fontsize=18)
            plt.title("F", fontsize=18)
            plt.xlabel("Iteration", fontsize=18)
            plt.legend()
            plt.draw()

            filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_'+fun_name+'_eng_addit.pdf')
            plt.savefig(filename, format='pdf')

    #plt.show()

def plot_last_points(res_all, cfg):
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']
    markers = ['o', '^', 'd', (5,1)]
    labels = ["Arora", "Random-rare", "Random-uniform", "Clust-words", "SVD", "Clust-tfIdf"]
    for i, fun_name in enumerate(cfg['measure'].split(',')):
        points_med = [[] for i in xrange(6)]
        points_max = [[] for i in xrange(6)]
        points_min = [[] for i in xrange(6)]
        for results in res_all:
            plt.figure()
            val = np.array([r[:, i] for r in results])
            fun = getattr(measure, fun_name + '_name')
            plt.ylabel(fun(), fontsize=18)
            plt.title("F", fontsize=18)
            plt.xlabel(u"alpha", fontsize=18)
            #plt.grid(True)
            index_exp_series = 0
            for it, expirement_runs in enumerate([int(x) for x in cfg['runs'].split(",")]):
                series_stats = calculate_stats(val[index_exp_series:index_exp_series+expirement_runs, 0:], cfg['begin_graph_iter'])
                points_med[it].append(series_stats[0][-1])
                points_max[it].append(series_stats[2][-1])
                points_min[it].append(series_stats[3][-1])
                #plt.plot(range(cfg['begin_graph_iter'], cfg['begin_graph_iter'] + len(series_stats[0])), series_stats[0], linewidth=2, c=colors[it % len(colors)], label = labels[it])
                #plt.fill_between(range(cfg['begin_graph_iter'], cfg['begin_graph_iter'] + len(series_stats[0])), series_stats[2], series_stats[3], alpha = 0.1, facecolor=colors[it % len(colors)])
                index_exp_series += expirement_runs
        for ii in xrange(6):
            plt.plot([0.0,0.2,0.4,0.6,0.8,1.0], points_med[ii], linewidth=2, c=colors[ii % len(colors)], label = labels[ii])
            plt.fill_between([0.0,0.2,0.4,0.6,0.8,1.0], points_max[ii], points_min[ii], alpha = 0.1, facecolor=colors[ii % len(colors)])
        plt.legend()
        plt.draw()

        filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_'+fun_name+'.pdf')
        plt.savefig(filename, format='pdf')
        fun = getattr(measure, fun_name + '_name_eng')
        plt.ylabel(fun(), fontsize=18)
        plt.title("F", fontsize=18)
        plt.xlabel("Iteration", fontsize=18)
        plt.legend()
        plt.draw()

        filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_'+fun_name+'_eng.pdf')
        plt.savefig(filename, format='pdf')


def ivan_experiment():
    for T in [70,100]:
        load_data = [2,5,4,1,1,3,3]
        result_dir = ['test/024_04_16_random_'+str(T)+'_/', 'test/024_04_16_random_uniform'+str(T)+'_/', 'test/024_04_16_diag_'+str(T)+'_/','test/024_04_16_nips_'+str(T)+'_/','test/024_04_16_kos_'+str(T)+'_/','test/024_04_16_nips_halfsint_'+str(T)+'_','test/024_04_16_kos_halfsint_'+str(T)+'_']
        data_name = ['','','','nips','kos','nips','kos']
        for i, l in enumerate(load_data):
            cfg = config.load('config.txt')
            cfg['T'] = T
            cfg['load_data'] = l
            cfg['result_dir'] = result_dir[i]
            cfg['data_name'] = data_name[i]
            cfg['compare_real'] = 1
            if l == 1:
                cfg['compare_real'] = 0
                cfg['save_matrices'] = '0, 0, 1, 0, 0, 0'
            else:
                cfg['save_matrices'] = '0, 0, 0, 0, 0, 0'
            if l == 3:
                cfg['seed'] = 446
            else:
                cfg['seed'] = 444
            if l in [1,3] and T == 70:
                continue
            if l != 3:
                if not os.path.exists(cfg['result_dir']):
                    os.makedirs(cfg['result_dir'])
                results, finals = main.main(cfg=cfg)
                plot_results(results, finals, cfg)
            else:
                last_points_results = []
                last_points_finals = []
                for alpha in [0.0,0.2,0.4,0.6,0.8,1.0]:
                    cfg['result_dir'] = result_dir[i]+str(alpha)+"/"
                    if not os.path.exists(cfg['result_dir']):
                        os.makedirs(cfg['result_dir'])
                    cfg['alpha'] = alpha
                    results, finals = main.main(cfg=cfg)
                    plot_results(results, finals, cfg)
                    last_points_results.append(results)
                cfg['result_dir'] = result_dir[i]
                if not os.path.exists(cfg['result_dir']):
                    os.makedirs(cfg['result_dir'])
                plot_last_points(last_points_results, cfg)


if __name__ == '__main__':
    ivan_experiment()
