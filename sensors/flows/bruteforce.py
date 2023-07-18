from sensors.flows.lazy import LazyEvaluator

class BruteForceEvaluator(LazyEvaluator):
    def __init__(self, G, labeled_edges):
        super().__init__(G, labeled_edges)
        self.candidates = list(labeled_edges.keys())
        
    def pop(self, cores=8):
        with Pool(cores) as pool:
            errs = pool.map(lambda e: (self.evaluate(self.sensors + [e]), e), self.candidates)
            s = min(errs, key=lambda x: x[0])
            
        self.candidates.remove(s[1])
        self.sensors.append(s[1])
        return s[1]