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
#     for p in ["small", "sdc", "lac", "sfc"]:
#         print("running sensor selection on {}".format(p))
#         run_experiment(p)
#         print()
        
    tntp_files = [
        "data/Anaheim/Anaheim_flow.tntp",
        "data/Barcelona/Barcelona_flow.tntp",
        "data/Chicago-Sketch/ChicagoSketch_flow.tntp",
        "data/Winnipeg/Winnipeg_flow.tntp"
    ]
    
    tntp_prefixes = ["Anaheim", "Barcelona", "Chicago", "Winnipeg"]
    
    for filename, prefix in zip(tntp_files, tntp_prefixes):
        print("running sensor selection on {}".format(prefix))
        run_tntp_experiment(filename, prefix)
        
if __name__ == "__main__":
    main()