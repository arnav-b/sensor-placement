import numpy as np
import networkx as nx
from scipy.sparse import linalg
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csr_matrix, issparse
from scipy.optimize import fmin_l_bfgs_b
import scipy
from tqdm.notebook import tqdm
from multiprocess import Pool 
import heapq
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans

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

def speed_prediction(G, labelled_flows, lamb=1e-6):
    '''
        Speed prediction.
    '''
    n_labelled_edges = len(labelled_flows)
    line_G = nx.line_graph(G)
    A = nx.adjacency_matrix(line_G)
    f0 = []
    
    A.data = A.data / np.repeat(np.add.reduceat(A.data, A.indptr[:-1]), np.diff(A.indptr))
   
    U = []
    L = []
    i = 0
    j = 0
    index = {}
    for e in line_G.nodes():
        if e in labelled_flows:
            f0.append(labelled_flows[e])
            L.append(i)
        else:
            U.append(i)
            index[e] = j
            j = j + 1
        i = i + 1

    PUL = A[np.ix_(U, L)]
    f0 = np.array(f0)
    b = PUL.dot(f0)
    PUU = A[np.ix_(U, U)]
    A = scipy.sparse.identity(PUU.shape[0])-PUU   
    res = linalg.lsmr(A, b, damp=lamb)
    x = res[0]
    
    pred_flows = {}

    for e in G.edges:
        if e in labelled_flows:
            pred_flows[e] = labelled_flows[e]
        else:
            pred_flows[e] = x[index[e]]
            
    return pred_flows

def dict2vect(G, labels):
    x = np.zeros(G.number_of_edges())
    for i, e in enumerate(G.edges()):
        try:
            x[i] = labels[e]
        except KeyError:
            continue
    return x

def vect2dict(G, x):
    labels = {}
    for i, e in enumerate(G.edges()):
        labels[e] = x[i]
    return labels
    

def mse(pred_labels, labeled_edges):
    return sum([(v - pred_labels[k]) ** 2 for k, v in labeled_edges.items()]) / len(labeled_edges)

def choose_sensors(G, labeled_edges, predict, evaluate, k=None, lazy=True, cores=8, debug=False):
    """
    Choose `k` sensors greedily to optimize the prediction given by `predict` for ground truth `labels`
    based on loss function `evaluate` (lower loss is better). 
    
    Parameters:
        -- G: networkx graph
        -- labeled_edges: dict {edge : value}
        -- predict: function taking G, labeled_edges -> prediction
        -- evaluate: prediction, true_values -> loss
    """
    if k == None:
        k = int(G.number_of_edges() / 50)
    
    sensors = []
        
    # Find value of adding each edge not in the set
    current = evaluate(predict(G, {}), labeled_edges)

    with Pool(cores) as pool:
        deltas = pool.map(lambda e: (evaluate(predict(G, {e : labeled_edges[e]}), labeled_edges) - current, e), 
                          labeled_edges)

    heapq.heapify(deltas)
    
    if debug:
        print("initial len(deltas):", len(deltas))
    
    # Greedily select sensors
    if lazy:
        for i in tqdm(range(k)):
            current = evaluate(predict(G, {e: labeled_edges[e] for e in sensors}), labeled_edges)
            if debug:
                print("iteration {}, current {}".format(i, current))

            # Current top 
            _, s = heapq.heappop(deltas)
            delta = evaluate(predict(G, {e: labeled_edges[e] for e in sensors} | {s : labeled_edges[s]}), 
                             labeled_edges) - current
            
            if len(deltas) == 0:
                sensors.append(s)
                continue

            # Next top
            delta_next, s_next = heapq.heappop(deltas)
            delta_next -= current
            
            if debug:
                print("sensor {} delta {}".format(s, delta))
                print("next top sensor {} delta {}".format(s_next, delta_next))
            
            if delta <= delta_next:
                heapq.heappush(deltas, (delta_next, s_next))
                
            # If the change drops it below the next best, recalculate the next best and continue
            while delta > delta_next:
                if debug:
                    print("sensor {} delta {}".format(s, delta))
                    print("next top sensor {} delta {}".format(s_next, delta_next))
                delta_next = evaluate(predict(G, {e: labeled_edges[e] for e in sensors} | {s_next : labeled_edges[s_next]}), 
                             labeled_edges) - current
                if delta_next < delta:
                    heapq.heappush(deltas, (delta, s))
                    delta, s = delta_next, s_next
                    if debug:
                        print("next > delta, len(deltas)", len(deltas))
                else:
                    heapq.heappush(deltas, (delta_next, s_next))
                    if debug:
                        print("delta > next, len(deltas):", len(deltas))
                    delta_next, s_next = heapq.heappop(deltas)
                    
                    if delta <= delta_next:
                        heapq.heappush(deltas, (delta_next, s_next))
                        break
            if debug:    
                print("iteration", i, "len(deltas):", len(deltas))
                print("selecting sensor {} delta {}, next sensor {} has delta {}".format(s, delta, s_next, delta_next))
                print()

            sensors.append(s)
    else:
        for i in tqdm(range(k)):
            current = evaluate(predict(G, {e: labeled_edges[e] for e in sensors}), labeled_edges)
            if debug:
                print("iteration {}, current {}".format(i, current))
            opt, opt_cost = None, float("inf")
            for s in G.edges():
                if s not in sensors:
                    cost = evaluate(predict(G, {e : labeled_edges[e] for e in sensors} | {s : labeled_edges[s]}),
                                   labeled_edges)
                    if debug:
                        print("sensor {} cost {} opt {}".format(s, cost, opt_cost))
                    if cost < opt_cost:
                        opt, opt_cost = s, cost
            sensors.append(opt)
        
    return sensors

def choose_sensors_random(G, k=None, dummy=None):
    """
    Randomly select k edges from the graph.
    """
    if k == None:
        k = int(G.number_of_edges() / 50)
        
    H = G.__class__()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges)
    
    if dummy in H.nodes:
        H.remove_node(dummy)
        
    choice = np.random.choice(H.number_of_edges(), k, replace=False)
    selected = []
    for i, e in enumerate(H.edges):
        if i in choice:
            selected.append(e)
        
    return selected

def synthetic_flows(G, b=0.02, epsilon=0.1):
    B = nx.incidence_matrix(G, oriented=True)
    u, s, vh = scipy.linalg.svd(B.todense())
    ss = np.zeros(vh.shape[0])
    ss[:s.shape[0]] = s
    
    vec = np.zeros(vh.shape[0])

    for i in range(vh.shape[0]):
        vec = vec + b / (ss[i]+epsilon) * vh[i]
        
    flows = {}
    
    i = 0
    for e in G.edges():
        flows[e] = vec[i] * 1000
        i = i + 1
        
    return flows

def geh(pred, actual):
    if pred == actual == 0:
        return np.array([0])
    return np.sqrt((2 * (pred - actual) ** 2) / (pred + actual))

def mape(pred_labels, labeled_edges):
    err = 0
    for k, v in labeled_edges.items():
        if abs(v) > 1e-10:
            err += 100 * abs((v - pred_labels[k]) / v)
    return err

def al_flows_rrqr(G, ratio):
    B = nx.incidence_matrix(G, oriented=True).todense()
    VC = scipy.linalg.null_space(B)
    q, r, p = scipy.linalg.qr(VC.T, pivoting=True)
    edges = list(G.edges())
    selected = []

    for e in list(p[0:int(ratio*G.number_of_edges())]):
        selected.append(edges[e])
        
    return selected

def al_flows_rb(G, ratio, n_dim=2):
    A = nx.adjacency_matrix(G).todense()
    nodes = list(G.nodes())
    embedding = SpectralEmbedding(n_components=n_dim, affinity='precomputed')
    spec_emb = embedding.fit_transform(A)
    selected = []
    
    clusters = [list(np.arange(0,G.number_of_nodes()))]
    
    while len(selected) < int(ratio*G.number_of_edges()):
        max_cluster = 0
        
        for i in range(len(clusters)):
            if len(clusters[i]) > len(clusters[max_cluster]):
                max_cluster = i
        
        cluster = clusters[max_cluster]
        
        if len(cluster) <= 2:
            break
            
        X = spec_emb[cluster,:]
                
        labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X).labels_
        clusters[max_cluster] = []
        clusters.append([])
        
        for i in range(len(cluster)):
            if labels[i] == 0:
                clusters[-1].append(cluster[i])
            else:
                clusters[-2].append(cluster[i])
        
        for v in clusters[-1]:
            for u in clusters[-2]:
                if G.has_edge(nodes[v], nodes[u]):
                    selected.append((nodes[v], nodes[u]))
                
                    if len(selected) >= int(ratio*G.number_of_edges()):
                        break
                        
    return selected