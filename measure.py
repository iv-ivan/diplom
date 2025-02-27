#!/usr/bin/python2
# -*- coding: utf-8 -*-

import config

from numpy import *
def normalize_cols(matrix):
    return matrix / maximum(sum(matrix, 0), 1e-7)
# metrics

def perplexity_old(A, B):
    return exp(-sum(A * log(maximum(B, 1e-7))) / maximum(sum(A), 1e-7))

def perplexity(A, B):
    return exp(-sum(A * log(maximum(B, 1e-7))) / maximum(sum(A), 1e-7))

def perplexity_name():
    return u'Перплексия'

def perplexity_name_eng():
    return 'Perplexity'

def frobenius(A, B):
    Z = copy(B)
    for i in xrange(A.shape[1]):
        s = sum(A[:,i])
        Z[:, i] = Z[:,i]*s
    return sqrt(sum((A - Z) ** 2))

def frobenius_old(A, B):
   return sqrt(sum((A - B) ** 2))

def rmse(A, B):
    return sqrt(mean((A - B) ** 2))


def rmse_name():
    return u'Корень среднеквадратичной ошибки'


def frobenius_name():
    return u'Норма Фробениуса'

def frobenius_name_eng():
    return 'Frobenius norm'

def kl(A, B):
    return sum(A * log(maximum(A / maximum(B, 1e-7), 1e-7)) - A + B)


def kl_name():
    return u'KL дивергенция'


def hellinger(A, B):
    A_scaled = normalize_cols(A)
    res = 0.
    for i in xrange(A.shape[1]):
        res += sqrt(sum((sqrt(A_scaled[:,i]) - sqrt(B[:,i])) ** 2) / 2)
    return res / A.shape[1]


def hellinger_name():
    return u'Расстояние Хеллингера'

def hellinger_name_eng():
    return 'Hellinger distance'
# final measures of factorization

# common

def old_pmi(W, H, top_words=10):
    N, T = W.shape
    M = H.shape[1]
    pt = mean(H, axis=1)
    pw = sum(W * tile(pt, (N, 1)), axis=1)
    pmi_res = zeros((T, top_words, top_words), dtype='float64')
    for topic in xrange(T):
        twords = argsort(-W[:, topic])[:top_words]
        for wi, word in enumerate(twords):
            Wp = W[twords, :] * tile(W[word, :], (top_words, 1)) * pt
            pmi_res[topic, wi, :] = log(maximum(sum(Wp, axis=1) / (pw[twords] * pw[word]), 1e-7))
    return pmi_res

def pmi(F, W, H, top_words=10):
    N, T = W.shape
    M = H.shape[1]
    n_d = sum(F, axis=0)
    N = sum(F)
    pmi_res = zeros((T, top_words, top_words), dtype='float64')
    P = sum(F, axis = 1) / N
    for topic in xrange(T):
        twords = argsort(-W[:, topic])[:top_words]
        for wi, word in enumerate(twords):
            Wp = F[twords, :] * tile(F[word, :], (top_words, 1)) / (n_d*N)
            pmi_res[topic, wi, :] = log(maximum(sum(Wp, axis=1) / (P[twords] * P[word]), 1e-7))
    return pmi_res

def get_closest(W):
    T = W.shape[1]
    closest = zeros((T,))
    dis = zeros((T,))
    for topic in xrange(T):
        p = sqrt(W) - sqrt(tile(W[:, topic], (T, 1))).T
        res = sqrt(sum(p ** 2, axis=0) / 2)
        res[topic] = res.max()
        closest[topic] = argmin(res)
        dis[topic] = res.min()
    return closest, dis


def cl_cov(W):
    closest, _ = get_closest(W)
    T = W.shape[1]
    res = zeros((T,))
    for topic in xrange(T):
        res[topic] = cov(vstack((W[:, topic].T, W[:, closest[topic]].T)))[0, 1]
    return res

# special

def mean_pmi(F, W, H):
    N, T = W.shape
    P = pmi(F, W, H)
    #for topic in xrange(T):
    #    res += mean(P[topic, :, :])
    return ('Mean PMI', mean(P))


def mean_min_pmi(W, H):
    N, T = W.shape
    P = pmi(W, H)
    res = zeros((T,))
    for topic in xrange(T):
        #print(P[topic, :, :])
        #print(P[topic, :, :].min)
        #print(min(P[topic, :, :]))
        res[topic] = P[topic, :, :].min()
    return 'Mean Min PMI', mean(res)


def mean_max_pmi(W, H):
    N, T = W.shape
    P = pmi(W, H)
    res = zeros((T,))
    for topic in xrange(T):
        res[topic] = P[topic, :, :].max()
    return 'Mean Max PMI', mean(res)


def var_mean_pmi(W, H):
    N, T = W.shape
    P = pmi(W, H)
    res = zeros((T,))
    for topic in xrange(T):
        res[topic] = P[topic, :, :].min()
    return ('Var Mean PMI', var(res))


def var_min_pmi(W, H):
    N, T = W.shape
    P = pmi(W, H)
    res = zeros((T,))
    for topic in xrange(T):
        res[topic] = P[topic, :, :].min()
    return ('Var Min PMI', var(res))


def mean_cov(W, H):
    C = cov(W.T)
    return 'Mean Cov', mean(C)


def var_cov(W, H):
    C = cov(W.T)
    return 'Var Cov', var(C)


def max_cl_cov(W, H):
    return 'Max Cl Cov', cl_cov(W).max()


def mean_cl_cov(W, H):
    return 'Mean Cl Cov', mean(cl_cov(W))


def mean_hell(W, H):
    T = W.shape[1]
    dis = zeros((T,T))
    for topic in xrange(T):
        p = sqrt(W) - sqrt(tile(W[:, topic], (T, 1))).T
        dis[topic, :] = sqrt(sum(p ** 2, axis=0) / 2)
    return 'Mean Hell', mean(dis)


def mean_nhell(F, W, H):
    _, dis = get_closest(W)
    return 'Mean NHell', mean(dis)


def min_nhell(W, H):
    _, dis = get_closest(W)
    return 'Min NHell', dis.min()
