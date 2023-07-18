import numpy as np
import pandas as pd
import networkx as nx
import pickle
import scipy

def read_traffic_data(name):
    G = nx.from_pandas_edgelist(pd.read_csv("data/{}.csv".format(name), header=None), source=0, target=1)
    
    with open("data/{}_flows.pkl".format(name), "rb") as f:
        flows = {(int(k[0]), int(k[1])) : v for k, v in pickle.load(f).items()}
        for u, v in G.edges():
            if (v, u) in flows:
                flows[(u, v)] = flows[(v, u)]
                del flows[(v, u)]
    
    with open("data/{}_speeds.pkl".format(name), "rb") as f:
        speeds = {(int(k[0]), int(k[1])) : v for k, v in pickle.load(f).items()}
        for u, v in G.edges():
            if (v, u) in speeds:
                speeds[(u, v)] = speeds[(v, u)]
                del speeds[(v, u)]
        
    return G, flows, speeds

def read_tntp_graph(filename):
    edgelist = pd.read_csv(filename, sep="\t")
    G = nx.from_pandas_edgelist(edgelist, source="From ", target="To ", edge_attr="Volume ",
                           create_using=nx.DiGraph)
    labeled_flows = {(u,v) : d["Volume "] for u,v,d in G.edges(data=True)}
    return G, labeled_flows

def synthetic_speeds(G):
    line_G = nx.line_graph(G.to_undirected())
    eigenvalues, eigenvectors = np.linalg.eigh(nx.normalized_laplacian_matrix(line_G).toarray())
    return dict(zip(line_G.nodes(), sum(1 / eigenvalues[i] * eigenvectors[:, i] for i in range(1, G.number_of_edges()))))

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