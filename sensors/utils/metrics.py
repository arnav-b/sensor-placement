import networkx as nx
import numpy as np

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
    for i, e in enumerate(nx.line_graph(G).nodes()):
        labels[e] = x[i]
    return labels

def mse(pred_labels, labeled_edges):
    return sum([(v - pred_labels[k]) ** 2 for k, v in labeled_edges.items()]) / len(labeled_edges)

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