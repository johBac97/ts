#!/bin/env python3

import numpy as np
from pathlib import Path
from loguru import logger
import itertools 
from argparse import ArgumentParser
import json

import matplotlib.pyplot as plt 
import matplotlib.animation as anim
import matplotlib.lines as lines


class Animator:
    def __init__(self, graph: np.ndarray, nodes: np.ndarray, solutions: np.ndarray):
        self._graph = graph
        self._solutions = solutions
        self._nodes = nodes

        self._text_offset = 0.005

        logger.debug(f"Graph:\t{self._graph}")
        logger.debug(f"Solutions shape:\t{self._solutions.shape}")
        logger.debug(f"Nodes:\t{self._nodes}")

    @classmethod
    def from_graph_and_log_files(self,graph_path: Path, log_path: Path):

        # First read graph file
        with open(graph_path, "r") as io:

            # First line contains number of nodes
            n_nodes = int(io.readline().strip())

            graph = np.zeros((0, n_nodes), dtype=np.int64)

            # Next lines contains the distance graph
            for line_id in range(0, n_nodes):
                values = io.readline().strip().split()
                array = np.array(values, dtype=np.float64)
                graph = np.r_[graph, array[np.newaxis, :]]

            # Then there is the maximum distance
            max_distance = float(io.readline().strip())

            # The last `n_nodes` lines contains the node positions
            nodes = np.zeros((0, 2), dtype=np.float64)

            for line_id in range(0, n_nodes):
                values = io.readline().strip().split()
                array = np.array(values, dtype=np.float64)
                nodes = np.r_[nodes, array[np.newaxis, :]]

        # Then read the log file
        with open(log_path, "r") as io:
            
            solutions = np.zeros((0, n_nodes + 1), dtype=np.int64)

            while line := io.readline().strip():
                if not "STATUS" in line:
                    continue
                data = json.loads(line.split("AlgorithmStatus:")[1].strip())
                array = np.array(data["best_chromosome"], dtype=np.int64)
                solutions = np.r_[solutions, array[np.newaxis]]
                

        return Animator(graph, nodes, solutions)

    def _anim_init(self):
        self._ax.set_xlim(0.0,1.0)
        self._ax.set_ylim(0.0,1.0)
        
        # Plot initial frame with nodes
        scatter_plt = self._ax.scatter(self._nodes[:,0], self._nodes[:,1])
        
        # Initialize an empty route plot
        self._route_plot = lines.Line2D([],[])

        self._ax.add_line(self._route_plot)


        # Add text to the nodes
        start_node_text = self._ax.text(self._nodes[0,0] + self._text_offset, self._nodes[0,1] + self._text_offset, "Start/Goal", fontsize=12, color="red")

        self._node_texts = []
        for (i,n) in enumerate(self._nodes[1:,:]):
            self._node_texts.append(self._ax.text(0,0, f"{i+1}", fontsize=12, color="red"))

        return self._route_plot,self._node_texts

    def _anim_data_gen(self):
        for t in itertools.count():
            yield t, self._solutions[t,:]

    def _anim_run(self, data):
        
        t, sol = data
        
        nodes = self._nodes[sol, :]

        logger.debug(f"Solution {t}:\t{sol}")
        logger.debug(f"Nodes {t}:\t{nodes}")

        self._route_plot.set_data(nodes[:,0], nodes[:,1]) 

        for (text, node) in zip(self._node_texts, nodes[1:-1,:]):
            text.set_position(node + self._text_offset)

        return self._route_plot,self._node_texts
    
    def animate(self):

        # Init the figure and the axes
        self._fig , self._ax = plt.subplots(figsize=(30,30))
        self._ax.grid()


        self._anim = anim.FuncAnimation(self._fig, self._anim_run, self._anim_data_gen, interval=1000, init_func=self._anim_init)

        plt.show()

        pass


def parse_args():

    parser = ArgumentParser(
        "Animation Generator\nCreate an animation from a log file from one of the Genetic Solvers."
    )

    parser.add_argument(
        "graph",
        type=Path,
        help="Path to the graph file defining the Traveling Salesman problem.",
    )

    parser.add_argument(
        "logfile",
        type=Path,
        help="Path to the log file containing the output from one of the Genetic Solvers.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    animator = Animator.from_graph_and_log_files(args.graph, args.logfile)

    animator.animate()
