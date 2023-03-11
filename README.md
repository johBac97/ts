# TravelingSalesmanGeneticSolver

Solvers for the [Traveling salesman
problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem) using a
[genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm). The repo
contain a Python3 and a C++ implementation of the same algorithm, along with
some helpful scripts to generate input graph files and for speed comparison.

### Background

This repo was created as an educational oppurtunity to explore genetic
algorithms as well as allow me(johBac97) to sharpen my C++ skills. An emphasis
has been placed on making the two implementations of the algorithm as
equivalent as possible to allow for an accurate speed comparison. 

### Building C++ version

The source files for the C++ implementation are stored in the `cpp/` folder.
The Makefile contained in the top of the repo allows for easy building of the
executable. Using the command,

``` 
make 
```

places a built executable `gen_solver` at the root of the repo. 

### Generating a graph file 

The algorithm an input graph file that describes the particular problem. To
generate such a file use the command:

```
python scripts/generate_graph.py --num-nodes NUM_NODES --max-distance
MAX_DISTANCE GRAPH_OUTPUT_PATH 
```

where `NUM_NODES` are the number of nodes (or cities) to be used when
generating this problem and `MAX_DISTANCE` is the maximum distance between any
two nodes. The generated graph is stored in `GRAPH_OUTPUT_PATH`.

### Usage

To run the python version use the command;

```
py/gen_solver.py PATH_TO_GRAPH
```

and to run the C++ version use the following command after the executable has been built;

```
./gen_solver PATH_TO_GRAPH
```

where `PATH_TO_GRAPH` is the path to the graph file defining the problem.

Optional arguments used by both implementations:

  - `--generations N`: Set the number of generations that the algorithm should run.
  - `--pop-size N`: Set the number of individuals in the population.
  - `--drop-frac R`: Set the fraction of lowest scoring individuals that should be dropped each generation.
  - `--mutation-frac R`: Set the fraction of individuals that should be mutated every generation.
  - `--no-print`: Disable printing to standard output.
  - `--output PATH`: Write the results of the algorithm to a text file.

