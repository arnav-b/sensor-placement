from sensors import utils
from sensors import flows

def run_experiment(prefix):
    G, labeled_flows, labeled_speeds = utils.data.read_traffic_data(prefix)
    evaluator = flows.lazy.LazyEvaluator(G, labeled_flows, debug=True)
    evaluator.write_results(prefix)
    
def run_tntp_experiment(filename, prefix):
    G, labeled_flows = utils.data.read_tntp_graph(filename)
    evaluator = flows.lazy.LazyEvaluator(G, labeled_flows, debug=True)
    evaluator.write_results(prefix)
    
def main():
    for p in ["small", "sdc", "lac", "sfc"]:
        print("running sensor selection on {}".format(p))
        run_experiment(p)
        print()
        
if __name__ == "__main__":
    main()