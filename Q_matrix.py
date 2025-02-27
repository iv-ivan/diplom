from __future__ import division
import numpy as np
import time
import scipy.sparse
import math
from  helper_functions import *

# Given a sparse CSC document matrix M (with floating point entries),
# comptues the word-word correlation matrix Q
def generate_Q_matrix(M, words_per_doc=None):
    
    simulation_start = time.time()
    
    vocabSize = M.shape[0]
    numdocs = M.shape[1]
    
    diag_M = np.zeros(vocabSize)
    
    if type(M) == scipy.sparse.csc.csc_matrix:
        sparse = 1
    else:
        sparse = 0
    
    if sparse:
        for j in xrange(M.indptr.size - 1):
            
            # start and end indices for column j
            start = M.indptr[j]
            end = M.indptr[j + 1]
            
            wpd = np.sum(M.data[start:end])
            if words_per_doc != None and wpd != words_per_doc:
                print 'Error: words per doc incorrect'
            
            row_indices = M.indices[start:end]
            
            diag_M[row_indices] = diag_M[row_indices] + M.data[start:end]/(wpd*(wpd-1))
            M.data[start:end] = M.data[start:end]/math.sqrt(wpd*(wpd-1))
    else:
        for j in xrange(numdocs):
            wpd = np.sum(M[:, j])
            if wpd < 1:
                continue
            #    wpd = 1 + 1e-6
            diag_M = diag_M + M[:, j] / (wpd * (wpd - 1))
            M[:, j] = M[:, j] / np.sqrt(wpd * (wpd - 1))
    
    Q = np.dot(M, M.T) / numdocs
    if sparse:
        Q = Q.toarray()
    else:
        Q = np.array(Q, copy=False)
    #print 'Sum of entries in Q is ', np.sum(Q)

    diag_M = diag_M/numdocs
    Q = Q - np.diag(diag_M)
    
    print 'Sum of entries in Q is ', np.sum(Q)
    print 'Sum of entries in diag_M is', np.sum(diag_M)
    print 'Multiplying Q took ', str(time.time() - simulation_start), 'seconds'
    
    return Q
