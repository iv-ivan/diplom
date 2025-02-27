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
       real_phi_sparsity
       gen_theta
       real_theta_sparsity
    """
    N = cfg['N']
    T_0 = cfg['T_0']
    M = cfg['M']

    gen_phi = getattr(generators, cfg['gen_phi'])
    cfg['rows'] = N
    cfg['cols'] = T_0
    cfg['sparsity'] = cfg['real_phi_sparsity']
    Phi_r = gen_phi(cfg)

    gen_theta = getattr(generators, cfg['gen_theta'])
    cfg['rows'] = T_0
    cfg['cols'] = M
    cfg['sparsity'] = cfg['real_theta_sparsity']
    Theta_r = gen_theta(cfg)

    F = np.dot(Phi_r, Theta_r)
    for i in xrange(F.shape[1]):
        F[:, i] = F[:,i] * np.random.randint(100,8000)
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
    #F_norm = normalize_cols(F)
    T = Theta.shape[0]
    eps = cfg['eps']
    schedule = cfg['schedule'].split(',')
    meas = cfg['measure'].split(',')
    val = np.zeros((cfg['max_iter']+2, len(meas)))
    hdist = np.zeros((2, cfg['max_iter']+2))#Phi - first row, Theta - second
    
    for i, fun_name in enumerate(meas):
        fun = getattr(measure, fun_name)
        val[0, i] = fun(F, np.dot(Phi, Theta))
    
    if cfg['compare_real']:
        #m = Munkres()
        idx = get_permute(Phi_r, Theta_r, Phi, Theta, cfg['munkres'])
        hdist[0][0] = hellinger(Phi[:, idx[:, 1]], Phi_r[:, idx[:, 0]])
        hdist[1][0] = hellinger(Theta[idx[:, 1],:], Theta_r[idx[:, 0],:])

    if cfg['print_lvl'] > 1:
        print('Initial loss:', val[0])
    status = 0
    methods_num = len(schedule)
    it = -1
    for it in range(cfg['max_iter']+1):
        if cfg['print_lvl'] > 1:
            print('Iteration', it+1)
        ####Phi_old = deepcopy(Phi)
        ####Theta_old = deepcopy(Theta)
        method_name = schedule[it % methods_num]
        if cfg['print_lvl'] > 1:
            print('Method:', method_name)
        method = getattr(methods, method_name)
        (Phi, Theta) = method(F, Phi, Theta, method_name, cfg)
        #jogging of weights
        if cfg['jogging'] == 1 and it < 10:
            joh_alpha = 0.25
            cfg['phi_sparsity'] = 0.05
            cfg['theta_sparsity'] = 0.1
            Phi_jog, Theta_jog = gen_init(cfg)
            Phi = (1-joh_alpha**(it+1))*Phi + joh_alpha**(it+1)*Phi_jog
            Theta = (1-joh_alpha**(it+1))*Theta + joh_alpha**(it+1)*Theta_jog
        for j, fun_name in enumerate(meas):
            fun = getattr(measure, fun_name)
            val[it+1, j] = fun(F, np.dot(Phi, Theta))#fun(F_norm, np.dot(Phi, Theta))
        
        if cfg['compare_real']:
            idx = get_permute(Phi_r, Theta_r, Phi, Theta, cfg['munkres'])
            hdist[0][it+1] = hellinger(Phi[:, idx[:, 1]], Phi_r[:, idx[:, 0]])
            hdist[1][it+1] = hellinger(Theta[idx[:, 1], :], Theta_r[idx[:, 0], :])
        
        if cfg['print_lvl'] > 1:
            print(val[it+1])
        if all(val[it, :] < eps):
            if cfg['print_lvl'] > 1:
                print('By cost.')
            status = 1
            break
        '''if abs(Phi_old - Phi).max() < eps and abs(Theta_old - Theta).max() < eps:
            if cfg['print_lvl'] > 1:
                print('By argument.')
            status = 2
            break'''
        #del W_old
        #del H_old
    if cfg['print_lvl'] > 1:
        print('Final:')
    #Phi = normalize_cols(Phi)
    #Theta = normalize_cols(Theta)
    #for j, fun_name in enumerate(meas):
    #    fun = getattr(measure, fun_name)
    #    val[it+2:, j] = fun(F, np.dot(Phi, Theta))#fun(F_norm, np.dot(Phi, Theta))
    
    #if cfg['compare_real']:
    #    idx = get_permute(Phi_r, Theta_r, Phi, Theta, cfg['munkres'])
    #    hdist[0][it+2:] = hellinger(Phi[:, idx[:, 1]], Phi_r[:, idx[:, 0]])
    #    hdist[1][it+2:] = hellinger(Theta[idx[:, 1],:], Theta_r[idx[:, 0], :])

    return (val, hdist, it, Phi, Theta, status)

def merge_halfmodel(F, Phi_r, Theta_r, cfg):
	F_model = np.dot(np.dot(Phi_r, Theta_r), np.diag(np.sum(F, axis=0)))
	alpha = cfg["alpha"]
	return F*alpha + F_model*(1-alpha)


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
        cfg['N'], cfg['M'] = F.shape
        print('Dimensions of F:', N, M)
        print('Checking assumption on F:', np.sum(F, axis=0).max())
        return F, vocab, N, M, None, None
    elif cfg['load_data'] == 2:
        F, Phi_r, Theta_r = gen_real(cfg)
        print(Phi_r)
        print('Checking assumption on F:', np.sum(F, axis=0).max())
        return F, None, F.shape[0], F.shape[1], Phi_r, Theta_r
    elif cfg['load_data'] == 3:
    	print("uci halfmodel", cfg["alpha"])
        F, vocab = data.load_uci(cfg['data_name'], cfg)
        N, M = F.shape
        cfg['N'], cfg['M'] = F.shape
        Phi_r, Theta_r = load_obj('Phi_'+cfg['data_name']), load_obj('Theta_'+cfg['data_name'])
        F_merged = merge_halfmodel(F, Phi_r, Theta_r, cfg)
        print('Dimensions of F:', N, M)
        print('Checking assumption on F:', np.sum(F_merged, axis=0).max())
        return F_merged, vocab, N, M, Phi_r, Theta_r
    elif cfg['load_data'] == 4:
        F = np.eye(cfg['T'])
        cfg['N'], cfg['M'] = F.shape
        Phi_r = np.eye(cfg['T'])
        Theta_r = np.eye(cfg['T'])
        return F, None, cfg['T'], cfg['T'], Phi_r, Theta_r
    elif cfg['load_data'] == 5:
        cfg['real_theta_sparsity'] = 1.
        cfg['real_phi_sparsity'] = 1.
        F, Phi_r, Theta_r = gen_real(cfg)
        print('Checking assumption on F:', np.sum(F, axis=0).max())
        return F, None, F.shape[0], F.shape[1], Phi_r, Theta_r

def construct_from_svd(U, s, V, cfg):
    T = cfg['T']
    Phi = np.zeros((U.shape[0], T))
    Theta = np.zeros((T, V.shape[1]))
    for i in xrange(T):
        x = U[:, i]
        y = V[i, :]
        xp = np.copy(x)
        xp[xp < 0] = 0
        xn = (-1)*np.copy(x)
        xn[xn < 0] = 0
        yp = np.copy(y)
        yp[yp < 0] = 0
        yn = (-1)*np.copy(y)
        yn[yn < 0] = 0
        xp_norm = np.linalg.norm(xp, ord=1)
        yp_norm = np.linalg.norm(yp, ord=1)
        xn_norm = np.linalg.norm(xn, ord=1)
        yn_norm = np.linalg.norm(yn, ord=1)
        if xp_norm*yp_norm > xn_norm*yn_norm:
            Phi[:, i] = np.sqrt(s[i]*xp_norm*yp_norm)*xp/xp_norm
            Theta[i, :] = np.sqrt(s[i]*xp_norm*yp_norm)*yp/yp_norm
        else:
            Phi[:, i] = np.sqrt(s[i]*xn_norm*yn_norm)*xn/xn_norm
            Theta[i, :] = np.sqrt(s[i]*xn_norm*yn_norm)*yn/yn_norm
    return normalize_cols(Phi), normalize_cols(Theta)

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
        F_norm = normalize_cols(F)
        Phi = prepare.anchor_words(F_norm, 'L2', cfg)
        print('Solving for Theta')
        Theta = np.linalg.solve(np.dot(Phi.T, Phi) + np.eye(Phi.shape[1]) * eps, np.dot(Phi.T, F_norm))
        Theta[Theta < eps] = 0
        Theta = normalize_cols(Theta)
        return Phi, Theta
    elif (int(cfg['prepare_method'].split(',')[i]) == 2):
        print("Random rare")
        cfg['phi_sparsity'] = 0.05
        cfg['theta_sparsity'] = 0.1
        return gen_init(cfg)
    elif (int(cfg['prepare_method'].split(',')[i]) == 3):
        print("Random uniform")
        cfg['phi_sparsity'] = 1.
        cfg['theta_sparsity'] = 1.
        return gen_init(cfg)
    elif (int(cfg['prepare_method'].split(',')[i]) == 4):
        eps = cfg['eps']
        F_norm = normalize_cols(F)
        print("Clustering of words")
        centroids, labels = prepare.reduce_cluster(F_norm, cfg['T'], cfg)
        Theta = centroids
        Theta[Theta < eps] = 0
        Theta = normalize_cols(Theta)
        print('Solving for Phi')
        Phi = np.transpose(np.linalg.solve(np.dot(Theta, Theta.T) + np.eye((Theta.T).shape[1]) * eps, np.dot(Theta, F_norm.T)))
        Phi[Phi < eps] = 0
        Phi = normalize_cols(Phi)
        return Phi, Theta
    elif (int(cfg['prepare_method'].split(',')[i]) == 5):
        eps = cfg['eps']
        F_norm = normalize_cols(F)
        print("SVD init")
        U, s, V = np.linalg.svd(F_norm)
        Phi, Theta = construct_from_svd(U, s, V, cfg)
        return Phi, Theta
    elif (int(cfg['prepare_method'].split(',')[i]) == 6):
        eps = cfg['eps']
        transformer = TfidfTransformer()
        transformer.fit(F)
        F_tfidf = (transformer.transform(F)).toarray()
        print("Clustering of tf-idf")
        centroids, labels = prepare.reduce_cluster(F_tfidf, cfg['T'], cfg)
        Theta = centroids
        Theta[Theta < eps] = 0
        Theta = normalize_cols(Theta)
        print('Solving for Phi')
        Phi = np.transpose(np.linalg.solve(np.dot(Theta, Theta.T) + np.eye((Theta.T).shape[1]) * eps, np.dot(Theta, F_tfidf.T)))
        Phi[Phi < eps] = 0
        Phi = normalize_cols(Phi)
        return Phi, Theta
    elif (int(cfg['prepare_method'].split(',')[i]) == 7):
        eps = cfg['eps']
        F_norm = normalize_cols(F)
        print("Clustering of words mixed")
        centroids, labels = prepare.reduce_cluster(F_norm, cfg['T'], cfg)
        Theta = centroids
        Theta[Theta < eps] = 0
        Theta = normalize_cols(Theta)
        print('Solving for Phi')
        Phi = np.transpose(np.linalg.solve(np.dot(Theta, Theta.T) + np.eye((Theta.T).shape[1]) * eps, np.dot(Theta, F_norm.T)))
        Phi[Phi < eps] = 0
        Phi = normalize_cols(Phi)
        cfg['phi_sparsity'] = 1.
        cfg['theta_sparsity'] = 1.
        Phi1, Theta1 = gen_init(cfg)
        zzz = 0.3
        return zzz*Phi1+(1.-zzz)*Phi, zzz*Theta1+(1.-zzz)*Theta
    elif (int(cfg['prepare_method'].split(',')[i]) == 8):
        print("Arora mixed")
        eps = cfg['eps']
        F_norm = normalize_cols(F)
        Phi = prepare.anchor_words(F_norm, 'L2', cfg)
        print('Solving for Theta')
        Theta = np.linalg.solve(np.dot(Phi.T, Phi) + np.eye(Phi.shape[1]) * eps, np.dot(Phi.T, F_norm))
        Theta[Theta < eps] = 0
        Theta = normalize_cols(Theta)
        cfg['phi_sparsity'] = 1.
        cfg['theta_sparsity'] = 1.
        Phi1, Theta1 = gen_init(cfg)
        zzz = 0.3
        return zzz*Phi1+(1.-zzz)*Phi, zzz*Theta1+(1.-zzz)*Theta
    elif (int(cfg['prepare_method'].split(',')[i]) == 9):
        print("Arora unifrom")
        eps = cfg['eps']
        F_norm = normalize_cols(F)
        Phi = prepare.anchor_words(F_norm, 'L2', cfg)
        print('Solving for Theta')
        Theta = np.ones((Phi.shape[1], F.shape[1]))
        Theta = normalize_cols(Theta)
        return Phi, Theta
    elif (int(cfg['prepare_method'].split(',')[i]) == 10):
        eps = cfg['eps']
        F_norm = normalize_cols(F)
        print("Clustering of docs")
        centroids, labels = prepare.reduce_cluster(F_norm.T, cfg['T'], cfg)
        Phi = centroids.T
        Phi[Phi < eps] = 0
        Phi = normalize_cols(Phi)
        print('Solving for Theta')
        Theta = np.linalg.solve(np.dot(Phi.T, Phi) + np.eye(Phi.shape[1]) * eps, np.dot(Phi.T, F_norm))
        Theta[Theta < eps] = 0
        Theta = normalize_cols(Theta)
        return Phi, Theta

def calculate_stats(series, begin_iter):
    series = series[:, begin_iter:]
    series_mean = np.mean(series, axis=0)
    series_var = np.var(series, axis=0)
    series_min = series[np.argmin(series[:,-1]),:]
    series_max = series[np.argmax(series[:,-1]),:]
    return series_mean, np.sqrt(series_var), series_min, series_max

def save_obj(obj, name):
    with open('./'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('./' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

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
       save_matrices
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
    finals = [[] for i in cfg['finals'].split(',')] # pmi, mean pmi etc
    prep_time = [0] * total_runs
    hdist_runs = [0] * total_runs #hellinger distances arrays
    exp_time = [0] * total_runs #time for run EM-algo
    general_info = []
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
    new_index = 0
    #general_info = load_obj(cfg['result_dir']+"general_info")
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
            prep_time[current_exp] = end_time
            #choose one method for compare (plsa)
            if cfg['compare_methods'] > 0:
                cfg['schedule'] = methods[0]

            #calculate usual EM-alg
            start = time()
            (val, hdist, i, Phi, Theta, status) = run(F, Phi, Theta, Phi_r, Theta_r, cfg)
            new_index +=1
            general_info.append((val, hdist, i, Phi, Theta, status))
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
                name, val = fun(F, Phi, Theta)
                finals[i].append(val)
                print(name, ':', val)

            #save results for different runs
            if int(cfg['save_matrices'].split(",")[it]) == 1 and r == 0:
            	save_obj(Phi, 'Phi_'+cfg['data_name'])
            	save_obj(Theta, 'Theta_'+cfg['data_name'])

            if cfg['save_topics']:
                visualize.save_topics(Phi, os.path.join(cfg['result_dir'], cfg['experiment'] + '_'+str(current_exp)+'topics.txt'), vocab)
            if cfg['compare_real']:
                pass#visualize.show_matrices_recovered(Phi_r, Theta_r, Phi, Theta, cfg, permute=True)
            current_exp += 1

    hdist_runs = np.array(hdist_runs)
    #save results section
    if cfg['experiment'] == '':
        exp_name = 'test'
    else:
        exp_name = cfg['experiment']

    labels = ["","Arora","Random-rare", "Random-uniform", "Clust-words", "SVD","Clust-tfidf", "Mixed-clust", "Mixed-Arora","Arora uniform","Clust-docs"]
    #save mean times
    index_exp_series = 0
    with open(os.path.join(cfg['result_dir'], cfg['experiment']+'_times.txt'),"w") as f:
        f.write("\\begin{tabular}{ |r | r | }\n\\hline\n\\multicolumn{2}{|c|}{Время работы алгоритмов} \\\\\n\\hline\n & Initialization & EM \\\\\n\\hline\n")
        for it, expirement_runs in enumerate([int(x) for x in cfg['runs'].split(",")]):
            cur_mean_exp_time = np.median(exp_time[index_exp_series:index_exp_series+expirement_runs])
            cur_mean_prep_time = np.median(prep_time[index_exp_series:index_exp_series+expirement_runs])
            f.write(labels[int(cfg['prepare_method'].split(',')[it])] + " & " + str(cur_mean_prep_time) + "&" + str(cur_mean_exp_time)+"\\\\\n")
            index_exp_series += expirement_runs
        f.write("\\hline\n\\end{tabular})")

    if cfg['compare_real']:
        index_exp_series = 0
        plt.figure()
        plt.title("Phi", fontsize=18)
        plt.ylabel(u"Расстояние Хеллингера", fontsize=18)
        plt.xlabel(u"Номер итерации", fontsize=18)
        #plt.grid(True)
        colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k', '#ffa000', '#7366bd']
        for it, expirement_runs in enumerate([int(x) for x in cfg['runs'].split(",")]):
            #Phi
            series_stats = calculate_stats(hdist_runs[index_exp_series:index_exp_series+expirement_runs, 0, 0:], cfg['begin_graph_iter'])
            plt.plot(range(cfg['begin_graph_iter'], cfg['begin_graph_iter'] + len(series_stats[0])), series_stats[0], linewidth=2, c=colors[it % len(colors)], label = labels[int(cfg['prepare_method'].split(',')[it])])
            plt.fill_between(range(cfg['begin_graph_iter'], cfg['begin_graph_iter'] + len(series_stats[0])), series_stats[2], series_stats[3], alpha = 0.1, facecolor=colors[it % len(colors)])
            index_exp_series += expirement_runs

        plt.legend()
        plt.draw()
        filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_Phi'+'.pdf')
        plt.savefig(filename, format='pdf')
        plt.ylabel("Hellinger distance", fontsize=18)
        plt.xlabel("Iteration", fontsize=18)
        plt.legend()
        plt.draw()
        filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_Phi_eng'+'.pdf')
        plt.savefig(filename, format='pdf')

        plt.figure()
        plt.title("Theta", fontsize=18)
        plt.ylabel(u"Расстояние Хеллингера", fontsize=18)
        plt.xlabel(u"Номер итерации", fontsize=18)
        #plt.grid(True)
        index_exp_series = 0

        for it, expirement_runs in enumerate([int(x) for x in cfg['runs'].split(",")]):
            #Theta
            series_stats = calculate_stats(hdist_runs[index_exp_series:index_exp_series+expirement_runs, 1, 0:], cfg['begin_graph_iter'])
            plt.plot(range(cfg['begin_graph_iter'], cfg['begin_graph_iter'] + len(series_stats[0])), series_stats[0], linewidth=2, c=colors[it % len(colors)], label = labels[int(cfg['prepare_method'].split(',')[it])])
            plt.fill_between(range(cfg['begin_graph_iter'], cfg['begin_graph_iter'] + len(series_stats[0])), series_stats[2], series_stats[3], alpha = 0.1, facecolor=colors[it % len(colors)])
            index_exp_series += expirement_runs

        plt.legend()
        plt.draw()
        filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_Theta'+'.pdf')
        plt.savefig(filename, format='pdf')
        plt.ylabel("Hellinger distance", fontsize=18)
        plt.xlabel("Iteration", fontsize=18)
        plt.legend()
        plt.draw()
        filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_Theta_eng'+'.pdf')
        plt.savefig(filename, format='pdf')

    save_obj(general_info, cfg['result_dir']+"general_info")
    return results, finals
    #plt.show()

if __name__ == '__main__':
    print("Loading config...")
    cfg = config.load()
    print("Config is loaded")
    if not os.path.exists(cfg['result_dir']):
        os.makedirs(cfg['result_dir'])

    print("Calculations:")
    results, finals = main(cfg=cfg)
    #save_obj(results, cfg['result_dir']+"res")
    #save_obj(finals, cfg['result_dir']+"fin")

    print("Plot graphs")
    colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k', '#ffa000', '#7366bd']
    markers = ['o', '^', 'd', (5,1)]
    labels = ["","Arora","Random-rare", "Random-uniform", "Clust-words", "SVD","Clust-tfidf", "Mixed-clust", "Mixed-Arora","Arora uniform","Clust-docs"]

    with open(os.path.join(cfg['result_dir'], cfg['experiment']+'_finals.txt'),"w") as f:   
        f.write("\\begin{tabular}{ |r | r r | }\n\\hline\n\\multicolumn{3}{|c|}{Метрики качества матрицы слова-темы $\Phi$}\\\\\\hline\n & Mean PMI  & Mean NHell \\\\\\hline\n")
        index_exp_series = 0    
        for it, expirement_runs in enumerate([int(x) for x in cfg['runs'].split(",")]):
            f.write(labels[int(cfg['prepare_method'].split(',')[it])])
            for i, fun_name in enumerate(cfg['finals'].split(',')):
                fun = getattr(measure, fun_name)
                name, val = fun(np.array([[1]]), np.array([[1]]), np.array([[1]]))
                series_mean = np.mean(finals[i][index_exp_series:index_exp_series+expirement_runs])
                series_max = np.max(finals[i][index_exp_series:index_exp_series+expirement_runs])
                series_min = np.min(finals[i][index_exp_series:index_exp_series+expirement_runs])
                #f.write(str(it+1)+" "+str(name)+" "+str(series_mean)+" "+str(series_max)+" "+str(series_min)+"\n")
                f.write(" & " + str(series_mean))
            f.write('\\\\\n')
            index_exp_series += expirement_runs
        f.write("\\hline\n\\end{tabular}")
    copyfile("config.txt", os.path.join(cfg['result_dir'], cfg['experiment']+'_config.txt'))

    for i, fun_name in enumerate(cfg['measure'].split(',')):
        plt.figure()
        val = np.array([r[:, i] for r in results])
        fun = getattr(measure, fun_name + '_name')
        plt.ylabel(fun(), fontsize=13)
        plt.title("F", fontsize=13)
        plt.xlabel(u"Номер итерации", fontsize=13)
        #plt.grid(True)
        index_exp_series = 0
        for it, expirement_runs in enumerate([int(x) for x in cfg['runs'].split(",")]):
            series_stats = calculate_stats(val[index_exp_series:index_exp_series+expirement_runs, 0:], cfg['begin_graph_iter'])
            plt.plot(range(cfg['begin_graph_iter'], cfg['begin_graph_iter'] + len(series_stats[0])), series_stats[0], linewidth=2, c=colors[it % len(colors)], label = labels[int(cfg['prepare_method'].split(',')[it])])
            plt.fill_between(range(cfg['begin_graph_iter'], cfg['begin_graph_iter'] + len(series_stats[0])), series_stats[2], series_stats[3], alpha = 0.1, facecolor=colors[it % len(colors)])
            index_exp_series += expirement_runs

        plt.legend()
        plt.draw()

        filename = os.path.join(cfg['result_dir'], cfg['experiment']+'_'+fun_name+'.pdf')
        plt.savefig(filename, format='pdf')
        fun = getattr(measure, fun_name + '_name_eng')
        plt.ylabel(fun(), fontsize=13)
        plt.title("F", fontsize=13)
        plt.xlabel("Iteration", fontsize=13)
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
                plt.plot(range(1, 1 + len(series_stats[0])), series_stats[0], linewidth=2, c=colors[it % len(colors)], label = labels[int(cfg['prepare_method'].split(',')[it])])
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
