# Sensor Placement

Simulations for the sensor placement project. Currently I only have results for the small L.A. network. Files produced are located in `out` divided by algorithm. The max flow solution is in `out/flowrouter` and the random sampling/ILP algorithm is in `out/routeSampler`. 

## How to Run

### Max Flow

Generate flows:

`python $SUMO_HOME/tools/detector/flowrouter.py --debug -i 60 -n out/small.net.xml -d out/small.detectors.xml -f out/small.readings.csv -o out/flowrouter/small.routes.xml -e out/flowrouter/small.flows.xml -c "qPKW" --revalidate-detectors --vclass passenger`

Run simulation:

`sumo -c flowrouter.sumocfg -v`

### Random Sampling

Generate random routes (flags/options control hyperparameters):

`python $SUMO_HOME/tools/randomTrips.py -n out/small.net.xml -r out/routeSampler/small.rou.xml --lanes --fringe-factor 10 --random-routing-factor=1.5 -e 86400`

Run the constraint solver:

`python $SUMO_HOME/tools/routeSampler.py -r out/routeSampler/small.rou.xml --edgedata-files out/routeSampler/small.edgeData.xml -o out/routeSampler/small.sampledRoutes.xml --optimize full`

Run simulation:

`sumo -c routeSampler.sumocfg -v`

## Results

Best results to date are in `out/routeSampler/small.edgedata.csv`. 
