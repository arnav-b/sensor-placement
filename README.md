# Sensor Placement

Simulations for the sensor placement project. Currently I only have results for the small L.A. network. Files produced are located in `out` divided by algorithm. The max flow solution is in `out/flowrouter` and the random sampling/ILP algorithm is in `out/routeSampler`. 

## How to Run

Max flow:

`sumo -c flowrouter.sumocfg -v`

Random sampling:

`sumo -c routeSampler.sumocfg -v`

## Results

Best results to date are in `out/routeSampler/small.edgedata.csv`. 
