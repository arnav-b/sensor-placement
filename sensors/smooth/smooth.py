import networkx as nx
import numpy as np
import pandas as pd
import scipy
import scipy.sparse.linalg as linalg
import pickle
import random
from scipy import sparse

from sensors import utils

class SmoothNetwork:
    def __init__(self, G, labeled_edges, labeled_flows=None, debug=False):
        self.G = G
        self.line_G = nx.line_graph(G.to_undirected())
        self.debug = debug
        self.P = sparse.csc_matrix(transition_matrix(self.line_G))
        self.m = self.line_G.number_of_nodes()

        self.node2idx = dict(zip(self.line_G.nodes(), range(self.m)))
        self.idx2node = dict(zip(range(self.m), self.line_G.nodes()))
        nx.relabel_nodes(self.line_G, self.node2idx, copy=False)

        self.labeled_edges = labeled_edges
        if labeled_flows:
            self.labeled_flows = labeled_flows
        else:
            self.labeled_flows = {e: 1 for e in labeled_edges}
            
        # Create true signal and weight vectors
        self.x = np.zeros(self.m)
        self.w = np.zeros(self.m)
        for e, s in self.labeled_edges.items():
            self.x[mapping[e]] = s 
            self.w[mapping[e]] = self.labeled_flows[e]
            
    def log(self, msg):
        if self.debug:
            print(msg)
            
    def predict_swap(self, i, v):
        S_prime = self.S[:i] + [v] + self.S[i+1:]
        swap = np.where(np.array(self.T) == v)[0][0]
        T_prime = self.T[:swap] + [self.S[i]] + self.T[swap+1:]

        P_TT_prime = self.P[np.ix_(T_prime, T_prime)]

        P_TS_prime = self.P[np.ix_(T_prime, S_prime)]
        x_S_prime = self.x[np.ix_(S_prime)]

        U = np.zeros((self.m-len(self.T), 2))
        U[:, 1] = self.P_TT.toarray()[:, swap] - P_TT_prime.toarray()[:, swap]
        U[swap] = [1, 0]

        V = np.zeros((2, m-len(self.T)))
        V[0] = P_TT.toarray()[swap] - P_TT_prime.toarray()[swap]
        V[:, swap] = [0, 1]

        x_int = lu.solve(P_TS_prime @ x_S_prime)
        Y = lu.solve(U)

        x_int = sparse.csc_array(x_int).transpose()
        Y = sparse.csc_matrix(Y)
        V = sparse.csc_matrix(V)
        I = sparse.identity(2, format="csc")
        x_T_prime_sparse = x_int - Y @ sparse.linalg.inv(I + V @ Y) @ V @ x_int        
    
    def al_swap(self, ratio):
        k = int(ratio * len(labeled_edges))
        
        # Choose random sensors
        C = [mapping[e] for e in labeled_edges]
        self.S = random.sample(C, k)
        self.T = sorted(list(set(line_G.nodes()) - set(S)))
        self.P_TT = P[np.ix_(T, T)]
        P_TS = P[np.ix_(T, S)]
        
        self.lu = sparse.linalg.splu(sparse.identity(self.m-k, format="csc") - P_TT)

        self.log("computed LU decomposition")
        
        x_S = x[np.ix_(self.S)]
        x_T_hat = lu.solve(P_TS @ x_S)
        
        current_err = weightedL2(x_T_hat, x[np.ix_(T)], w[np.ix_(T)])
        old_err = float("inf")
        iteration = 0
        
        self.log("beginnning search")
        
        while old_err - current_err > 1e5:
            self.log("iteration {} current error {:e}".format(iteration, current_err))
            
            for i in tqdm(range(k)):
                # Move sensor i to a more optimal placement
                candidates = random.sample(list(set(C).difference(set(S))), min(10, len(C) - len(S)))
                
                for v in candidates:
                    

                    # If error decreased, keep the swap
                    new_err = weightedL2(x_T_prime_sparse.transpose().toarray(), x[np.ix_(T_prime)], w[np.ix_(T_prime)])
                    if new_err < current_err:
                        if debug:
                            print("swapping nodes, recomputing LU decomposition...", end="")

                        # New sensor and target sets
                        current_err = new_err
                        S = S_prime
                        T = T_prime
                        P_TT = P_TT_prime

                        # Recompute LU decomposition
                        lu = sparse.linalg.splu(sparse.identity(m-k, format="csc") - P_TT)

                        if debug:
                            print("done")