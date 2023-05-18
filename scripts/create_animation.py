#!/bin/env python3

import numpy as np
from pathlib import Path
from loguru import logger
import itertools
from argparse import ArgumentParser, BooleanOptionalAction
import json

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.lines as lines

# Find path to ffmpeg binary
from subprocess import run, PIPE
proc = run("which ffmpeg", shell=True, stdout=PIPE)
if proc.returncode != 0:
    logger.warning("Unable to find ffmpeg. Saving animations will use slower Pillow library")
else:
    plt.rcParams['animation.ffmpeg_path'] = proc.stdout.decode("ascii").strip()


class Animator:
    def __init__(self, graph: np.ndarray, nodes: np.ndarray, solutions: np.ndarray):
        self._graph = graph
        self._solutions = solutions
        self._nodes = nodes

        self._text_offset = 0.005

        self._anim_settings = {"interval": 100, "repeat": True}

        self._general_settings = {"show": True, "save_path": None}

        #logger.debug(f"Graph:\t{self._graph}")
        #logger.debug(f"Solutions shape:\t{self._solutions.shape}")
        #logger.debug(f"Nodes:\t{self._nodes}")

    @classmethod
    def from_graph_and_log_files(self, graph_path: Path, log_path: Path):

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

    @property
    def settings(self):
        settings = {}

        settings.update(map(lambda k, v: ("anim_" + k, v), self._anim_settings.items()))
        settings.update(map(lambda k, v: ("general_" + k, v), self._general_settings.items()))

        return settings

    def set_settings(self, **kwargs):


        for (k, v) in kwargs.items():
            if k.startswith("anim_"):
                self._anim_settings[k.removeprefix("anim_")] = v
            elif k.startswith("general_"):
                # General settings
                self._general_settings[k.removeprefix("general_")] = v
            else:
                raise ValueException(f"Unkown setting:\t{k}")

    def _anim_init(self):

        self._ax.grid()
        self._ax.axis("off")
        self._ax.set_xlim(0.0, 1.0)
        self._ax.set_ylim(0.0, 1.0)

        # Plot initial frame with nodes
        scatter_plt = self._ax.scatter(self._nodes[:, 0], self._nodes[:, 1])

        # Initialize an empty route plot
        self._route_plot = lines.Line2D([], [])

        self._ax.add_line(self._route_plot)

        # Add text to the nodes
        start_node_text = self._ax.text(
            self._nodes[0, 0] + self._text_offset,
            self._nodes[0, 1] + self._text_offset,
            "Start/Goal",
            fontsize=12,
            color="red",
        )

        self._node_texts = []
        for (i, n) in enumerate(self._nodes[1:, :]):
            self._node_texts.append(
                self._ax.text(0, 0, f"{i+1}", fontsize=12, color="red")
            )

        return self._route_plot, self._node_texts

    def _anim_data_gen(self):
        for t in range(self._solutions.shape[0]):
            yield t, self._solutions[t, :]

    def _anim_run(self, data):

        t, sol = data
        self._ax.set_title(f"Generation {t}")

        nodes = self._nodes[sol, :]

        #logger.debug(f"Solution {t}:\t{sol}")
        #logger.debug(f"Nodes {t}:\t{nodes}")

        self._route_plot.set_data(nodes[:, 0], nodes[:, 1])

        for (text, node) in zip(self._node_texts, nodes[1:-1, :]):
            text.set_position(node + self._text_offset)

        return self._route_plot, self._node_texts

    def animate(self):

        logger.info("Starting animation...")

        # Init the figure and the axes
        self._fig, self._ax = plt.subplots(figsize=(30, 30))

        self._anim = anim.FuncAnimation(
            self._fig,
            self._anim_run,
            self._anim_data_gen,
            init_func=self._anim_init,
            **self._anim_settings,
        )

        logger.info("Completed!")

        if self._general_settings["show"]:
            logger.info("Showing Animation")
            plt.show()

        if self._general_settings["save_path"] is not None:
            logger.info(f"Saving animation:\t{self._general_settings['save_path']}")
            self._anim.save(self._general_settings["save_path"], fps=int(1/self._anim_settings["interval"]), dpi=70)

        logger.success("Finished")

def parse_args():

    parser = ArgumentParser(
        prog="Animation Generator",
        description="Create an animation from a log file from one of the Genetic Solvers.",
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

    parser.add_argument("--fps", type=int, help="Number of frames per second.", default=30)

    parser.add_argument("--output", type=Path, help="Path to save the animation to.")

    parser.add_argument(
        "--show",
        action=BooleanOptionalAction,
        help="Show the plot on screen when created.",
        default=True,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    animator = Animator.from_graph_and_log_files(args.graph, args.logfile)

    settings = {"anim_interval": 1 / args.fps, "general_show": args.show, "general_save_path": args.output}

    animator.set_settings(**settings)
    animator.animate()
