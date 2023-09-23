import networkx as nx
import scipy.linalg as linalg
from sensors import utils, flows
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsmr

from sensors.utils.metrics import mse, dict2vect, vect2dict
from sensors.flows.flowprediction import *

class NoisyEvaluator(sensors.flows.lazy.LazyEvaluator):
    
    def __init__(self, G, true_edges, noise=1.5):
        noisy_edges = {}
        sigma = noise * np.std(list(true_edges.values()))
        for e, f in true_edges.items():
            noisy_edges[e] = f + np.random.normal(0, sigma)
                
        super().__init__(G, noisy_edges)
        self.true_edges = true_edges 
        
    def denoise(self, alpha):
        B = nx.incidence_matrix(self.G, oriented=True)
        m = self.G.number_of_edges()
        X = alpha * B.T @ B + np.identity(m)
        f_noisy = dict2vect(self.G, self.labeled_edges)
        return vect2dict(self.G, lsmr(X, f_noisy)[0])
        
    def evaluate(self, sensors):
        return mse(flow_prediction(self.G, {e: self.labeled_edges[e] for e in sensors}), self.labeled_edges)
    
    def predict(self, sensors):
        return flows.flow_prediction(self.G, {s: self.true_edges[s] for s in sensors})
    
    def get_pred_correlation(self, sensors):
        pred = self.predict(sensors)
        return np.corrcoef([pred[e] for e in self.true_edges], 
                           [self.true_edges[e] for e in self.true_edges])[0][1]
            
    def al_flows_rrqr(self, ratio):
        f = dict2vect(self.G, self.labeled_edges)
        B = nx.incidence_matrix(self.G, oriented=True)
        V_C = linalg.null_space(B.todense())
        
        f_C = V_C @ V_C.T @ denoise(self.G, f, 1)
        
        Q, R, p = linalg.qr(V_C.T @ np.diag(f_C), pivoting=True)
        
        sensors = []
        edges = list(self.G.edges())
        for i in range(int(ratio * len(self.labeled_edges))):
            sensors.append(edges[p[i]])
        
        return sensors
    
    def get_results(self):
        results = super().get_results()
        noisy_corr = np.corrcoef(dict2vect(self.G, self.true_edges), 
                                 dict2vect(self.G, self.labeled_edges))[0][1]
        results["noisy"] = {i / 10: noisy_corr for i in range(11)}
        return results