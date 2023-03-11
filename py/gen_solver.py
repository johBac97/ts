#!/bin/env python3

import numpy as np
from typing import Optional
import time
from numpy.typing import ArrayLike as NDArray
from datetime import datetime
from pathlib import Path
import traceback as tb
from argparse import ArgumentParser

import sys
from loguru import logger


######## SOLVER #######


class TravelingSalesmanGeneticSolver:
    def __init__(
        self,
        graph: np.ndarray,
        pop_size: int = 100,
        drop_frac: float = 0.05,
        mutation_frac: float = 0.3,
        max_generations: int = 10,
        seed: int = 0,
        output: Path = None,
    ):
        self._graph = graph
        self._pop_size = pop_size
        self._num_nodes = graph.shape[0]
        self._crossover_algo = "OX1"
        self._drop_frac = drop_frac
        self._mutation_frac = mutation_frac
        self._max_generations = max_generations
        self._output = output

        self._population = None
        self._scores = None
        self._generation = 0

        # Setup random number generator
        self._rng = np.random.default_rng(seed=seed)

    def _initialize(self):

        # initialize array containign the population
        self._population = np.resize(
            np.arange(1, self._num_nodes, dtype=np.int64),
            (self._pop_size, self._num_nodes - 1),
        )

        list(map(self._rng.shuffle, self._population))

        self._population = np.c_[
            np.zeros((self._pop_size, 1), dtype=np.int64),
            self._population,
            np.zeros((self._pop_size, 1), dtype=np.int64),
        ]

        self._scores = np.zeros(self._pop_size)

        self._check_if_valid_chromosomes()

    def _calculate_fitness(self):

        for chromosome in range(self._pop_size):
            score = 0.0
            pos = self._population[chromosome, 0]

            for i in range(1, self._num_nodes + 1):
                score += self._graph[pos, self._population[chromosome, i]]
                pos = self._population[chromosome, i]

            self._scores[chromosome] = score

        # sort chromosomes
        indicies = np.argsort(self._scores)

        self._population = self._population[indicies, :]
        self._scores = self._scores[indicies]

    def _crossover(self):

        n_offsprings = self._pop_size - self._population.shape[0]

        if n_offsprings <= 0:
            return

        pairs = self._rng.integers(
            0, high=self._population.shape[0], size=(n_offsprings, 2)
        )

        offsprings = np.zeros((0, self._num_nodes - 1), dtype=np.int64)
        for off_ind in range(n_offsprings):
            if self._crossover_algo == "OX1":
                offspring = self.OX1_crossover(
                    self._population[pairs[off_ind, 0], 1:-1].copy(),
                    self._population[pairs[off_ind, 1], 1:-1].copy(),
                )[np.newaxis, :]
            else:
                raise NotImplementedError("Unknown crossover algorithm")
            offsprings = np.r_[offsprings, offspring]

        # Append starting and trailing zero columns
        offsprings = np.c_[
            np.zeros(n_offsprings, dtype=np.int64),
            offsprings,
            np.zeros(n_offsprings, dtype=np.int64),
        ]

        # Add these new chromosomes to the population
        self._population = np.r_[self._population, offsprings]

        # Add zero scores for these new individuals
        self._scores = np.r_[self._scores, np.zeros(n_offsprings)]

    def _drop_least_fit(self):

        n_drop = int(np.floor(self._drop_frac * self._pop_size))

        self._population = self._population[: (self._pop_size - n_drop), :]
        self._scores = self._scores[: (self._pop_size - n_drop)]

    def _mutation(self):

        n_mutations = int(np.floor(self._pop_size * self._mutation_frac))

        chromosome_inds = self._rng.integers(self._pop_size, size=n_mutations)
        swap_inds = self._rng.integers(1, self._num_nodes - 1, size=(n_mutations, 2))

        mut_chromosomes = self._population[chromosome_inds, :]
        mut_chromosomes[
            np.arange(n_mutations), (swap_inds[:, 0], swap_inds[:, 1])
        ] = mut_chromosomes[np.arange(n_mutations), (swap_inds[:, 1], swap_inds[:, 0])]

        self._population[chromosome_inds, :] = mut_chromosomes

    def OX1_crossover(self, parent_1: NDArray, parent_2: NDArray):
        """
        Davis' Order Crossover (OX1) Algorithm from https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm
        """

        assert parent_1.shape == parent_2.shape
        length = parent_1.shape[0]

        offspring = np.zeros_like(parent_1)
        co_1, co_2 = np.sort(self._rng.integers(length, size=2))

        offspring[co_1:co_2] = parent_1[co_1:co_2]

        ind = co_2
        for p2 in np.roll(parent_2, length - co_2):
            if p2 not in offspring:
                offspring[ind] = p2
                ind += 1
                if ind == length:
                    ind = 0

        return offspring

    def _check_if_valid_chromosomes(self):
        """
        Debug function to check if all chromosomes are valid.
        """

        for (c_i, c) in enumerate(self._population):
            if not np.unique(c).shape[0] == c.shape[0] - 1:
                logger.warning(f"Chromosome number {c_i} is invalid: {c}")
                tb.print_stack()
                sys.exit(1)

    def _log_settings(self):

        logger.info(f"Input graph:\n{graph}")

        logger.info(f"Number of graph nodes:\t{self._num_nodes}")
        logger.info(f"Population size:\t{self._pop_size}")
        logger.info(f"Generations:\t{self._max_generations}")
        logger.info(f"Drop fraction:\t{self._drop_frac}")
        logger.info(f"Mutation fraction:\t{self._mutation_frac}")

    def run(self):

        logger.info("Starting TravelingSalesmanGeneticSolver!")

        self._log_settings()

        start_time = time.time()

        logger.info("Initializing population...")

        self._initialize()

        logger.info("Running algorithm!")

        while self._generation < self._max_generations:
            self._generation += 1

            # Calculate fitness of current population
            self._calculate_fitness()

            logger.info(
                f"Generation {self._generation}, shortest distance: {self._scores[0]}"
            )

            self._drop_least_fit()

            self._crossover()

            self._mutation()

        self._calculate_fitness()

        end_time = time.time()

        exec_time = round(end_time - start_time, 5)

        logger.success(
            f"Algorithm completed in {exec_time} seconds, shortest distance found:\t{self._scores[0]}"
        )

        logger.info(f"Solution:\t{self._population[0,:]}")

        # Write solution to a separate file
        if self._output is not None:
            if not self._output.parent.exists():
                self._output.parent.mkdir(parents=True)

            logger.info(f"Writing results to:\t{self._output.resolve()}")

            with open(self._output, "w") as io:
                io.write("Solution: [ %s ]\n" % " ".join(map(str, self._population[0,:].tolist())))
                io.write("Duration: %5.5f\n" % exec_time)
                io.write("Distance: %d\n" % self._scores[0])

        return {"solution": self._population[0, :].tolist(), "time": exec_time}


########## UTILS ##########


def parse_args():

    parser = ArgumentParser("Traveling Salesman Genetic Solver")

    parser.add_argument("graph", type=Path, help="Path to a numpy saved graph matrix.")

    parser.add_argument("--pop-size", help="Population size", type=int, default=100)

    parser.add_argument(
        "--drop-frac",
        help="Fraction of individuals to drop every generation.",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--mutation-frac",
        help="Fraction of individuals that are mutated every generation.",
        default=0.1,
        type=float,
    )

    parser.add_argument(
        "--seed", help="Set seed of the random number generator.", default=42, type=int
    )

    parser.add_argument(
        "--generations",
        help="Number of generations to run algorithm.",
        default=100,
        type=int,
    )

    # current_time_str = datetime.now().strftime("%Y-%M-%d_%H-%m-%S")
    parser.add_argument(
        "--logfile",
        help="Path to output log file",
        default=Path("logs/{time}_py.log"),
        type=Path,
    )

    parser.add_argument(
        "--no-print", help="Disable printing of status to stdout.", action="store_true"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write output of algorithm to separate file.",
    )

    return parser.parse_args()


def load_text_graph_format(path: Path):

    with open(path, "r") as io:
        # First line contains number of nodes
        n_nodes = int(io.readline().strip())

        graph = np.zeros((0, n_nodes), dtype=np.int64)

        while values := io.readline().strip().split():
            array = np.array(values, dtype=np.int64)
            graph = np.r_[graph, array[np.newaxis, :]]

        assert graph.shape[0] == graph.shape[1], "Non-square graph read from text file."

    return graph


def load_graph(path: Path):
    """
    Loads a graph defining the Traveling Salesmans Problem. Supports numpy array save format or custom text file format.
    """

    try:
        graph = np.load(path)
    except ValueError as error:
        # Not a valid numpy save format. Try to load file as custom text file format.
        graph = load_text_graph_format(path)

    assert np.all(np.diag(graph) == 0), "Diagonal not zero in loaded array."

    if not np.allclose(graph, graph.T):
        logger.warning("Loaded graph is not symmetric.")

    return graph


if __name__ == "__main__":

    args = parse_args()

    graph = load_graph(args.graph)

    if args.no_print:
        logger.remove(0)

    if not args.logfile.parent.exists():
        args.logfile.parent.mkdir()

    logger.add(args.logfile)

    logger.info(f"Graph file loaded:\t{args.graph.resolve()}")

    solver = TravelingSalesmanGeneticSolver(
        graph,
        drop_frac=args.drop_frac,
        pop_size=args.pop_size,
        mutation_frac=args.mutation_frac,
        max_generations=args.generations,
        seed=args.seed,
        output=args.output,
    )

    result = solver.run()
