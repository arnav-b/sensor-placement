import numpy as np
import networkx as nx
from scipy.sparse import linalg
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csr_matrix, issparse
from scipy.optimize import fmin_l_bfgs_b
import scipy

def flow_prediction(G, labelled_flows, lamb=1e-6):
    '''
        Flow prediction.
    '''
    n_labelled_edges = len(labelled_flows)
    b = np.zeros(n_labelled_edges)
    B = nx.incidence_matrix(G, oriented=True)
    f0 = np.zeros(G.number_of_edges())

    i = 0
    j = 0
    U = []
    index = {}
    for e in G.edges():
        if e in labelled_flows:
            f0[i] = labelled_flows[e]
        else:
            U.append(i)
            index[e] = j
            j = j + 1
        i = i + 1

    b = -B.dot(f0)
 
    A = B[np.ix_(np.arange(B.shape[0]), U)]

    res = linalg.lsmr(A, b, damp=lamb)
    x = res[0]

    pred_flows = {}
    
    for e in G.edges():
        if e in labelled_flows:
            pred_flows[e] = labelled_flows[e]
        else:
            pred_flows[e] = x[index[e]]
    
    return pred_flows

# Author: Vlad Niculae 
#         Lars Buitinck 
# Author: Chih-Jen Lin, National Taiwan University (original projected gradient
#     NMF implementation)
# Author: Anthony Di Franco (original Python and NumPy port)
# License: BSD 3 clause

def safe_fro(X, squared=False):
    if issparse(X):
        nrm = np.sum(X.data ** 2)
    else:
        if hasattr(X, 'A'):
            X = X.A
        nrm = np.sum(X ** 2)
    return nrm if squared else np.sqrt(nrm)
     

# Authors: Mathieu Blondel, Vlad Niculae
def nls_lbfgs_b(X, Y, W_init=None, l1_reg=0, l2_reg=0, max_iter=5000, tol=1e-3, callback=None):
    """Non-negative least squares solver using L-BFGS-B.
        
    Solves for W in
    min 0.5 ||Y - XW||^2_F + + l1_reg * sum(W) + 0.5 * l2_reg * ||W||^2_F
    
    """
    n_samples, n_features = X.shape
    n_targets = Y.shape[1]
    G = safe_sparse_dot(X.T, X)
    Xy = safe_sparse_dot(X.T, Y)

    def f(w, *args):
        W = w.reshape((n_features, n_targets))
        diff = (safe_sparse_dot(X, W) - Y)
        diff = diff.A if hasattr(diff, 'A') else diff
        res = 0.5 * np.sum(diff ** 2)
        if l2_reg:
            res += 0.5 * l2_reg * np.sum(W ** 2)
        if l1_reg:
            res += l1_reg * np.sum(W)
        return res

    def fprime(w, *args):
        W = w.reshape((n_features, n_targets))
        #grad = (np.dot(G, W) - Xy).ravel()
        grad = (safe_sparse_dot(G, W) - Xy).ravel()
        if l2_reg:
            grad += l2_reg * w
        if l1_reg:
            grad += l1_reg
        return grad

    if W_init is None:
        W = np.zeros((n_features * n_targets,), dtype=np.float64)
    else:
        W = W_init.ravel().copy()
    W, residual, d = fmin_l_bfgs_b(
                f, x0=W, fprime=fprime, pgtol=tol,
                bounds=[(0, None)] * n_features * n_targets,
                maxiter=max_iter,
                callback=callback)
    
    # testing reveals that sometimes, very small negative values occur
    W[W < 0] = 0
    
    if l1_reg:
        residual -= l1_reg * np.sum(W)
    if l2_reg:
        residual -= 0.5 * l2_reg * np.sum(W ** 2)
    residual = np.sqrt(2 * residual)
    if d['warnflag'] > 0:
        print("L-BFGS-B failed to converge")
    
    return W.reshape((n_features, n_targets)), residual

def flow_prediction_nls(G, labelled_flows, lamb=1e-6):
    '''
        Flow prediction with non-negative least squares
    '''
    n_labelled_edges = len(labelled_flows)
    b = np.zeros(n_labelled_edges)
    B = nx.incidence_matrix(G, oriented=True)
    f0 = np.zeros(G.number_of_edges())

    i = 0
    j = 0
    U = []
    index = {}
    for e in G.edges():
        if e in labelled_flows:
            f0[i] = labelled_flows[e]
        else:
            U.append(i)
            index[e] = j
            j = j + 1
        i = i + 1

    b = -B.dot(f0)
 
    A = B[np.ix_(np.arange(B.shape[0]), U)]
    
    #X_wide = np.random.uniform(low=0, high=1, size=(50, 100))
    #Y_wide = np.random.uniform(low=0, high=1, size=(50, 1))
    #W, resid = nls_projgrad(X_wide, Y_wide)
    
    #res = linalg.lsmr(A, b, damp=lamb)
    A = A.tocsr()
    #b = scipy.sparse.csr_matrix(b.reshape(b.shape[0], 1))
    b = scipy.sparse.csr_matrix(b.reshape(b.shape[0],1))
    
    res = nls_lbfgs_b(A, b, l2_reg=lamb)
    x = res[0]

    pred_flows = {}
    
    for e in G.edges():
        if e in labelled_flows:
            pred_flows[e] = np.array([labelled_flows[e]])
        else:
            pred_flows[e] = x[index[e]]
    
            
    return pred_flows