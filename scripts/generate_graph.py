#!/bin/env python3

import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from loguru import logger

from scipy.spatial.distance import cdist


def generate_float_graph(outfile: Path, num_nodes: int, max_distance: int, seed: int):
    pass

    if not outfile.parent.exists():
        outfile.parent.mkdir(parents=True)

    rng = np.random.default_rng(seed=seed)

    # Idea: Generate 2D-positions for the number of requested nodes. Then calculate the distance between each node and store in graph

    # 2d nodes
    nodes = rng.uniform(size=(num_nodes, 2))

    # Calculate distance between nodes
    dists = cdist(nodes, nodes, metric="euclidean")

    graph = (dists * max_distance / np.sqrt(2)).round(decimals=4)

    # Make sure it is symmetric
    graph = (graph + graph.T) / 2

    graph_generator = map(lambda x: " ".join(map(str, x)) + "\n", graph.tolist())

    node_generator = map(
        lambda x: " ".join(map(str, x)) + "\n", nodes.round(decimals=5).tolist()
    )

    # Write to file
    with open(outfile, "w") as io:

        # First line in file contains number of nodes
        io.write(str(num_nodes) + "\n")

        # Use generator to write the text representation of the graph
        io.writelines(graph_generator)

        io.write(str(max_distance) + "\n")

        io.writelines(node_generator)


def generate_int_graph(outfile: Path, num_nodes: int, max_distance: int, seed: int):

    if not outfile.parent.exists():
        outfile.parent.mkdir(parents=True)

    rng = np.random.default_rng(seed=seed)

    graph = rng.integers(max_distance, size=(num_nodes, num_nodes))

    # Make symmetric
    graph = ((graph + graph.T) / 2).astype(np.int64)

    np.fill_diagonal(graph, 0)

    line_generator = map(lambda x: " ".join(map(str, x)) + "\n", graph.tolist())

    with open(outfile, "w") as io:

        # First line in file contains number of nodes
        io.write(str(num_nodes) + "\n")

        # Use generator to write the text representation of the graph
        io.writelines(line_generator)


def parse_args():

    parser = ArgumentParser("Graph Creator")

    parser.add_argument(
        "--num-nodes",
        type=int,
        default=10,
        help="Number of nodes in the created graph.",
    )

    parser.add_argument(
        "--max-distance",
        type=int,
        default=20,
        help="The maximum distance between two nodes",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to use for the number generator."
    )

    parser.add_argument(
        "outfile", type=Path, help="The location of the output graph file."
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = vars(parse_args())

    logger.info("Graph Creation Settings")
    for (k, v) in args.items():
        logger.info(f"{k}:\t{v}")

    generate_float_graph(**args)
