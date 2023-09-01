import numpy as np
import networkx as nx
from scipy.sparse import linalg
from tqdm import tqdm
import pickle
from multiprocess import Pool 
import heapq
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans
import random

from sensors.utils.metrics import mse, dict2vect
from sensors.flows.flowprediction import *

class LazyEvaluator:
    def __init__(self, G, labeled_edges, debug=False):
        self.G = G
        self.labeled_edges = labeled_edges
        self.sensors = []
        self.debug = debug
        self.deltas = []
                
    def set_sensors(self, sensors):
        self.sensors = sensors
        
    def debug_print(self, msg):
        if self.debug:
            print(msg)
        
    def evaluate(self, sensors):
        return mse(flow_prediction(self.G, {e: self.labeled_edges[e] for e in sensors}), self.labeled_edges)
        
    def pop(self):
        current = self.evaluate(self.sensors)

        # Current top 
        _, s = heapq.heappop(self.deltas)
        delta = self.evaluate(self.sensors + [s]) - current
            
        if len(self.deltas) == 0:
            self.sensors.append(s)
            return s

        # Next top
        delta_next, s_next = heapq.heappop(self.deltas)
            
        # If the change drops it below the next best, recalculate the next best and continue
        while delta > delta_next:
            delta_next = self.evaluate(self.sensors + [s_next]) - current

            if delta_next <= delta:
                heapq.heappush(self.deltas, (delta, s))
                delta, s = delta_next, s_next
            else:
                heapq.heappush(self.deltas, (delta_next, s_next))
                
            delta_next, s_next = heapq.heappop(self.deltas)
            
        heapq.heappush(self.deltas, (delta_next, s_next))
        self.sensors.append(s)
        return s, delta
    
    def al_flows_greedy(self, ratio):
        k = int(ratio * len(self.labeled_edges))
        
        if k <= len(self.sensors):
            return self.sensors[:k]
        
        self.debug_print("calculating initial errors")
        
        current = self.evaluate([])
        for e in tqdm(self.labeled_edges):
            self.deltas.append((self.evaluate([e]) - current, e))
            
        heapq.heapify(self.deltas)
        
        self.debug_print("choosing sensors")

        for i in range(len(self.sensors), k):
            s, delta = self.pop()
            self.debug_print("selected {} of {}, delta {}".format(i+1, k, delta))

        return self.sensors
    
    def al_flows_rrqr(self, ratio, weighted=True):
        B = nx.incidence_matrix(self.G, oriented=True).todense()
        VC = scipy.linalg.null_space(B)
        U, s, Vh = scipy.linalg.svd(B)
        f = dict2vect(self.G, self.labeled_edges)
        q, r, p = scipy.linalg.qr(Vh @ np.diag(f), pivoting=True)
        edges = list(self.G.edges())
        selected = []

        for e in list(p):
            if edges[e] in self.labeled_edges:
                selected.append(edges[e])
                
            if len(selected) >= ratio * len(self.labeled_edges):
                break

        return selected
        
    def al_flows_rb(self, ratio, n_dim=2):
        A = nx.adjacency_matrix(self.G.to_undirected()).todense()
        nodes = list(self.G.nodes())
        embedding = SpectralEmbedding(n_components=n_dim, affinity='precomputed')
        spec_emb = embedding.fit_transform(A)
        selected = []

        clusters = [list(np.arange(0,self.G.number_of_nodes()))]
        mark_selected = {}

        while len(selected) < int(ratio*len(self.labeled_edges)):
            max_cluster = 0

            for i in range(len(clusters)):
                if len(clusters[i]) > len(clusters[max_cluster]):
                    max_cluster = i

            cluster = clusters[max_cluster]

            if len(cluster) <= 2:
                clusters[max_cluster] = []
                clusters.append([])
                clusters.append([])

                clusters[-1].append(cluster[0])
                clusters[-2].append(cluster[1])
            else:    
                X = spec_emb[cluster,:]

                labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X).labels_

                clusters[max_cluster] = []
                clusters.append([])
                clusters.append([])

                for i in range(len(cluster)):
                    if labels[i] == 0:
                        clusters[-1].append(cluster[i])
                    else:
                        clusters[-2].append(cluster[i])

            for v in clusters[-1]:
                for u in clusters[-2]:
                    if self.G.has_edge(nodes[v], nodes[u]) and (nodes[v],nodes[u]) not in mark_selected:
                        selected.append((nodes[v], nodes[u]))
                        mark_selected[(nodes[v],nodes[u])] = True
                        if len(selected) >= int(ratio*len(self.labeled_edges)):
                            return selected

                    if self.G.has_edge(nodes[u], nodes[v]) and (nodes[u],nodes[v]) not in mark_selected:
                        selected.append((nodes[u], nodes[v]))
                        mark_selected[(nodes[u],nodes[v])] = True
                        if len(selected) >= int(ratio*len(self.labeled_edges)):
                            return selected

        return selected
        
    def al_flows_random(self, ratio):
        return random.sample(list(self.labeled_edges.keys()), int(ratio * len(self.labeled_edges)))
    
    def al_flows_max(self, ratio):
        k = int(ratio * len(self.labeled_edges))
        return sorted(self.labeled_edges, key=self.labeled_edges.get, reverse=True)[:k]
    
    def predict(self, sensors):
        return flow_prediction(self.G, {s: self.labeled_edges[s] for s in sensors})
    
    def get_pred_correlation(self, sensors):
        pred = self.predict(sensors)
        return np.corrcoef([pred[e] for e in self.labeled_edges], 
                           [self.labeled_edges[e] for e in self.labeled_edges])[0][1]
    
    def get_results(self, samples=20):
        rrqr_sensors, rrqr_res = self.al_flows_rrqr(1), {0: 0}
        rand_sensors, rand_res = self.al_flows_random(1), {0: 0}
        greedy_sensors, greedy_res = self.al_flows_greedy(1), {0: 0}
        
        rb_sensors, rb_res = [], {}
        if len(self.labeled_edges) == self.G.number_of_edges():
            rb_sensors, rb_res = self.al_flows_rb(1), {0: 0}
            
        max_sensors, max_res = self.al_flows_max(1), {0: 0}
        
        self.debug_print("running experiment")
        
        for i in tqdm(range(1, samples + 1)):
            k =  int(i * len(self.labeled_edges) / samples)
            greedy_res[k / len(self.labeled_edges)] = self.get_pred_correlation(greedy_sensors[:k])
            rrqr_res[k / len(self.labeled_edges)] = self.get_pred_correlation(rrqr_sensors[:k])
            rand_res[k / len(self.labeled_edges)] = self.get_pred_correlation(rand_sensors[:k])
            max_res[k / len(self.labeled_edges)] = self.get_pred_correlation(max_sensors[:k])
            
            if rb_sensors:
                rb_res[k / len(self.labeled_edges)] = self.get_pred_correlation(rb_sensors[:k])
            
        return {
            "greedy": greedy_res,
            "rrqr": rrqr_res,
            "random": rand_res,
            "rb": rb_res,
            "max": max_res
        }
    
    def write_results(self, prefix):
        res = self.get_results()
        with open("results/{}_flow_correlations.pkl".format(prefix), "wb") as f:
            pickle.dump(res, f)
            
        with open("results/{}_flow_sensors.pkl".format(prefix), "wb") as f:
            pickle.dump(self.sensors, f)