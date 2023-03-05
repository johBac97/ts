
#include<vector>
#include<random>
#include<string>
#include<filesystem>


#ifndef TS_GENETIC_SOLVER_H
#define TS_GENETIC_SOLVER_H

// Define a struct containing a chromosome and its score.
struct Chromosome {
    std::vector<long> data;
    long score;
};

struct TravelingSalesmanGeneticSolverArgs{
    unsigned max_generations = 30;
    unsigned long pop_size = 100;
    double drop_frac = 0.2;
    double mutation_frac = 0.2;
    std::string crossover_algo = "OX1";
    unsigned seed = 42;
    std::filesystem::path output = std::filesystem::path();
};


class TravelingSalesmanGeneticSolver {
  private:
      std::vector<std::vector<long>> graph;
      unsigned max_generations;
      unsigned long num_nodes;
      unsigned long pop_size;
      unsigned long n_exchange; 
      unsigned seed;
      double drop_frac;
      double mutation_frac; 
      std::string crossover_algo;

      std::mt19937 engine;
      std::uniform_int_distribution<long> chromosome_index_dist;

      std::vector<Chromosome> population;

      std::filesystem::path output_file_path;


      void log_settings();
     
      // Initializes random chromosomes
      void initialize();
      
      // Core Genetic Algorithm functions
      void calculate_fitness(); 
      void drop_least_fit();
      void crossover();
      void mutation();

      void crossover_OX1(Chromosome, Chromosome, Chromosome&);

  public:
      TravelingSalesmanGeneticSolver(std::vector<std::vector<long>> graph, TravelingSalesmanGeneticSolverArgs = TravelingSalesmanGeneticSolverArgs());
      void run();
      void print_graph();
      void print_population();

};

#endif
