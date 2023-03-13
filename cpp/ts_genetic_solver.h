
#include<vector>
#include<random>
#include<string>
#include<iostream>
#include<filesystem>


#ifndef TS_GENETIC_SOLVER_H
#define TS_GENETIC_SOLVER_H

namespace src = boost::log::sources;

#define NUM_SEVERITY_LEVELS (5)

enum severity_level {
    status,
    info,
    success,
    warning,
    error
};


template< typename CharT, typename TraitsT > std::basic_ostream< CharT, TraitsT >&
operator<< (
          std::basic_ostream< CharT, TraitsT >& strm,
            severity_level lvl
        )
{
        static const char* severity_level_str[]  = {
            "STATUS",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR"
        };

        const char* str = severity_level_str[lvl];
        if (lvl < 5 && lvl >= 0)
             strm << str;
        else
             strm << lvl;
        return strm;
}



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
    bool log_progress = false;
};


class TravelingSalesmanGeneticSolver {
  private:
      std::vector<std::vector<long>> graph;
      unsigned max_generations;
      unsigned generation;
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
      bool log_status;

      src::severity_logger< severity_level > logger;

      void log_settings();
     
      // Initializes random chromosomes
      void initialize();
      
      // Core Genetic Algorithm functions
      void calculate_fitness(); 
      void drop_least_fit();
      void crossover();
      void mutation();

      void crossover_OX1(Chromosome, Chromosome, Chromosome&);

      void log_algo_status();

  public:
      TravelingSalesmanGeneticSolver(std::vector<std::vector<long>> graph, TravelingSalesmanGeneticSolverArgs = TravelingSalesmanGeneticSolverArgs());
      void run();
      void print_graph();
      void print_population();

};


#endif
