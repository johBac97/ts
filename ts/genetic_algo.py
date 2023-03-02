import numpy as np
from typing import Optional
import time
from numpy.typing import ArrayLike as NDArray
from pathlib import Path
import traceback as tb
from argparse import ArgumentParser

import sys
from loguru import logger

graph_1 = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])  # 5
graph_2 = np.array(
    [
        [0, 1, 2, 3, 4],
        [1, 0, 3, 2, 3],
        [2, 3, 0, 1, 3],
        [3, 2, 1, 0, 3],
        [4, 3, 3, 3, 0],
    ]
)  # 10


def OX1_crossover(parent_1: NDArray, parent_2: NDArray):
    """
    Davis' Order Crossover (OX1) Algorithm from https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm
    """

    assert parent_1.shape == parent_2.shape
    length = parent_1.shape[0]

    offspring = np.zeros_like(parent_1)
    co_1, co_2 = np.sort(np.random.randint(length, size=2))

    offspring[co_1:co_2] = parent_1[co_1:co_2]

    ind = co_2
    for p2 in np.roll(parent_2, length - co_2):
        if p2 not in offspring:
            offspring[ind] = p2
            ind += 1
            if ind == length:
                ind = 0

    return offspring


class TravelingSalesmanGeneticSolver:
    def __init__(
        self,
        graph: np.ndarray,
        pop_size: int = 100,
        drop_frac: float = 0.05,
        mutation_frac: float = 0.3,
        max_generations: int = 10,
    ):
        self._graph = graph
        self._pop_size = pop_size
        self._num_nodes = graph.shape[0]
        self._crossover_algo = "OX1"
        self._drop_frac = drop_frac
        self._mutation_frac = mutation_frac
        self._max_generations = max_generations

        self._population = None
        self._scores = None
        self._generation = 0

    def _initialize(self):

        # initialize array containign the population
        self._population = np.resize(
            np.arange(1, self._num_nodes, dtype=np.int64),
            (self._pop_size, self._num_nodes - 1),
        ).T

        np.random.shuffle(self._population)
        self._population = self._population.T

        self._population = np.c_[
            np.zeros((self._pop_size, 1), dtype=np.int64),
            self._population,
            np.zeros((self._pop_size, 1), dtype=np.int64),
        ]

        self._scores = np.zeros(self._pop_size)

    def _calculate_fitness(self):

        # TODO - First implementation does this sequentially very inefficient
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

        pairs = np.random.randint(
            0, high=self._population.shape[0], size=(n_offsprings, 2)
        )

        # TODO - optimize this shit
        offsprings = np.zeros((0, self._num_nodes - 1), dtype=np.int64)
        for off_ind in range(n_offsprings):
            if self._crossover_algo == "OX1":
                offspring = OX1_crossover(
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

        chromosome_inds = np.random.randint(self._pop_size, size=n_mutations)
        swap_inds = np.random.randint(1, self._num_nodes - 1, size=(n_mutations, 2))

        mut_chromosomes = self._population[chromosome_inds, :]
        mut_chromosomes[
            np.arange(n_mutations), (swap_inds[:, 0], swap_inds[:, 1])
        ] = mut_chromosomes[np.arange(n_mutations), (swap_inds[:, 1], swap_inds[:, 0])]

        self._population[chromosome_inds, :] = mut_chromosomes

    def _check_if_valid_chromosomes(self):

        # TODO very inefficient
        for (c_i, c) in enumerate(self._population):
            if not np.unique(c).shape[0] == c.shape[0] - 1:
                logger.warning(f"Chromosome number {c_i} is invalid: {c}")
                tb.print_stack()
                sys.exit(1)

    def run(self):

        start_time = time.time()

        self._initialize()

        while self._generation < self._max_generations:
            self._generation += 1

            # Calculate fitness of current population
            self._calculate_fitness()

            self._check_if_valid_chromosomes()

            logger.info(
                f"Generation {self._generation}, shortest distance: {self._scores[0]}"
            )

            self._drop_least_fit()

            self._check_if_valid_chromosomes()

            self._crossover()

            self._check_if_valid_chromosomes()

            self._mutation()

            self._check_if_valid_chromosomes()

        self._calculate_fitness()

        end_time = time.time()

        exec_time = round(end_time - start_time, 2)

        logger.success(
            f"Algorithm completed in {exec_time} second, best solution: {self._population[0,:]}"
        )


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

    return parser.parse_args()


def load_graph(path: str):
    graph = np.load(path)

    assert np.all(np.diag(graph) == 0), "Diagonal not zero in loaded array."

    if not np.allclose(graph, graph.T):
        logger.warning("Loaded graph is not symmetric.")

    return graph


if __name__ == "__main__":

    args = parse_args()

    np.random.seed(args.seed)

    graph = load_graph(args.graph)

    print(graph)

    solver = TravelingSalesmanGeneticSolver(
        graph,
        drop_frac=args.drop_frac,
        pop_size=args.pop_size,
        mutation_frac=args.mutation_frac,
        max_generations=args.generations,
    )

    solver.run()
