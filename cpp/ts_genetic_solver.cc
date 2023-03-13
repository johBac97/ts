

#include<boost/log/trivial.hpp>

#include<cstdlib>
#include<sstream>
#include<fstream>
#include<chrono>
#include<iostream>
#include<tuple>
#include<cmath>
#include<algorithm>
#include<random>

#include "ts_genetic_solver.h"

void print_progress_bar(double progress, int length = 120) {

    int pos;
    std::cout << "[";
    if (progress <= 1.0)
        pos = length * progress;
    else
        pos = length; 

    for(int i = 0;  i < length; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] ";
    if (progress <= 1.0)
        std::cout << int(progress * 100.0) << " %\r";
    else
        std::cout << "Done!" << std::endl;

    std::cout.flush();
}

TravelingSalesmanGeneticSolver::TravelingSalesmanGeneticSolver(std::vector<std::vector<long>> graph, TravelingSalesmanGeneticSolverArgs args)
{
    this->num_nodes = graph.size();
    this->graph = graph;

    this->max_generations = args.max_generations;
    this->pop_size = args.pop_size; this->drop_frac = args.drop_frac;
    this->mutation_frac = args.mutation_frac;
    this->crossover_algo = args.crossover_algo;

    this->seed = args.seed; // Only saved so that it can be logged later
    this->engine = std::mt19937(args.seed);

    // Number of chromosomes to be exchanged every generation
    this->n_exchange = ceil(this->drop_frac * this->pop_size);

    this->chromosome_index_dist = std::uniform_int_distribution<long>(1, this->num_nodes - 1);

    this->output_file_path = args.output;
    this->log_status = args.log_progress;

    this->generation = 0;
}

void TravelingSalesmanGeneticSolver::print_graph() 
{
    for (unsigned iy = 0; iy < num_nodes; iy++) {
        for (unsigned ix = 0; ix < num_nodes; ix++) {
          std::cout << graph[iy][ix] << " ";
        }
        std::cout << std::endl;
    }
}

void TravelingSalesmanGeneticSolver::print_population()
{
    for (auto c: this->population){
        for(auto i: c.data) 
            std::cout << i << " ";
        std::cout << ": " << c.score << std::endl;
    }
}


void TravelingSalesmanGeneticSolver::calculate_fitness() 
{

    // Calculate fitness of current population
    unsigned current_pos, next_pos;
    long score;
    std::vector<long> chr;

    for (unsigned chr_idx = 0; chr_idx < this->pop_size; chr_idx++){

        chr = this->population[chr_idx].data;
        current_pos = 0;
        score = 0;

        for (unsigned idx = 1; idx < this->num_nodes + 1; idx++){
            next_pos = chr[idx];
            score += this->graph[current_pos][next_pos];
            current_pos = next_pos;
        }
        this->population[chr_idx].score = score;

    }

    std::sort(this->population.begin(), this->population.end(), [](Chromosome a, Chromosome b)  -> bool {return a.score < b.score;});

}


void TravelingSalesmanGeneticSolver::drop_least_fit() 
{
    long n_drop;

    n_drop = this->n_exchange;
    this->population.erase(this->population.end() - n_drop , this->population.end());
}


void TravelingSalesmanGeneticSolver::initialize() 
{
  /*
   * Randomly initialize the population.
   */

  std::vector<long> temp(this->num_nodes + 1);
  for (unsigned pop_idx = 0; pop_idx < this->pop_size; pop_idx++){
      // Create one new random chromosome
    
      std::iota(std::begin(temp) + 1, std::end(temp) - 1, 1);

      std::shuffle(std::begin(temp) + 1 , std::end(temp) - 1, this->engine);

      // Add this random chromosome to the population
      this->population.push_back(Chromosome{temp, 0});
      temp.clear();
      temp.resize(this->num_nodes + 1);

  }
}


void TravelingSalesmanGeneticSolver::crossover() 
{
  unsigned long n_offsprings;
  Chromosome offspring;

  n_offsprings = this->n_exchange;

  // Randomly draw n_offsprings number of pairs which are to be the parents
  std::vector<std::tuple<long, long>> pairs(n_offsprings);

  std::uniform_int_distribution<long> dist(0, this->population.size() - 1);

  // Draw random pairs of indicies of the current population 
  std::generate(pairs.begin(),pairs.end(), [ this, &dist]() -> std::tuple<long, long> {
      return {dist(engine), dist(engine)};
  });

  // For every pair create a crossover offspring and add it to the population
  offspring.score = 0;
  for (auto p: pairs){
      auto p1 = this->population[std::get<0>(p)];
      auto p2 = this->population[std::get<1>(p)];

      offspring.data.clear();
      offspring.data.resize(this->num_nodes + 1);
      if (this->crossover_algo == "OX1"){
          crossover_OX1(p1, p2, offspring);
      }else{
          throw std::invalid_argument("Unkown crossover algorithm: " + this->crossover_algo + "\n");
      }

      this->population.push_back(offspring);
  }
}


void TravelingSalesmanGeneticSolver::crossover_OX1(Chromosome p1, Chromosome p2 , Chromosome& c)
{
    /*
      Davis' Order Crossover (OX1) Algorithm from https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm
    */
    long a,b,ind_1, ind_2;

    a = this->chromosome_index_dist(this->engine);
    b = this->chromosome_index_dist(this->engine);

    if (a > b)
        std::swap(a,b);

    // Copy segment between indicies from parent 1 to the offspring
    std::copy(p1.data.begin() + a, p1.data.begin() + b + 1, c.data.begin() + a);

    if (b + 1 == static_cast<long>(this->num_nodes))
        b = 0;

    ind_1 = b + 1; // Where to insert in offspring
    ind_2 = b + 1; // Which gene to pick from parent 2
    do {
        if (std::find(c.data.begin() , c.data.end(), p2.data[ind_2]) == c.data.end()){
            // The gene was not found in the offspring, insert it
            c.data[ind_1] = p2.data[ind_2];
            ind_1++;
            if (ind_1 >= static_cast<long>(this->num_nodes))
                ind_1 = 1;
        }
        ind_2++;
        if (ind_2 >= static_cast<long>(this->num_nodes))
            ind_2 = 1;

    } while (ind_2 != b + 1);
}

void TravelingSalesmanGeneticSolver::mutation()
{
    long n_mutations;
    long chr_index, gene_ind_1, gene_ind_2;

    n_mutations = ceil(this->mutation_frac * this->population.size());

    // Create distribution to draw which chromosomes should be mutated
    std::uniform_int_distribution<long> dist(0, this->population.size() - 1);

    std::vector<std::tuple<long,long,long>> mutation_indices(n_mutations);  

    // Draw indices
    std::generate(mutation_indices.begin(), mutation_indices.end() , [this, &dist]() -> std::tuple<long,long,long> {
            return {dist(engine), chromosome_index_dist(engine), chromosome_index_dist(engine)};
      });

    for(auto i: mutation_indices){
        std::tie(chr_index, gene_ind_1, gene_ind_2) = i;
        std::swap(this->population[chr_index].data[gene_ind_1] , this->population[chr_index].data[gene_ind_2]);
    }

}

void TravelingSalesmanGeneticSolver::log_settings()
{
    BOOST_LOG_SEV(logger, info) << "Settings:";
    BOOST_LOG_SEV(logger, info) << "Generations:\t" << this->max_generations;
    BOOST_LOG_SEV(logger, info) << "Population size:\t" << this->pop_size;
    BOOST_LOG_SEV(logger, info) << "Drop fraction:\t" << this->drop_frac;
    BOOST_LOG_SEV(logger, info) << "Mutation frac:\t" << this->mutation_frac;
    BOOST_LOG_SEV(logger, info) << "Seed:\t" << this->seed;
}

void TravelingSalesmanGeneticSolver::log_algo_status()
{
    // Log Generation id, current best score and current best solution
    
    std::string log_msg = "AlgorithmStatus:\t{";

    log_msg += "\"generation\":" + std::to_string(this->generation) + ",";
    log_msg += "\"min_distance\":" + std::to_string(this->population[0].score) + ",";
    log_msg += "\"best_chromosome\": [ ";

    std::stringstream best_chr_str;
    std::copy(this->population[0].data.begin(), this->population[0].data.end(), std::ostream_iterator<long>(best_chr_str, ", "));

    log_msg += best_chr_str.str();

    log_msg += "]}";

    BOOST_LOG_SEV(logger, status) << log_msg;
}

void TravelingSalesmanGeneticSolver::run() 
{

    BOOST_LOG_SEV(logger, info) << "Running TravelingSalesmanGeneticSolver on graph with " << this->num_nodes << " nodes.";

    log_settings();
    
    // Measure algorithm execution time
    auto start_time = std::chrono::high_resolution_clock::now();

    initialize();

    BOOST_LOG_SEV(logger, info) << "Running...";
    
    for(generation = 1; generation < this->max_generations ; generation++){

        if ((generation % (this->max_generations / 100)) == 0)
            print_progress_bar(static_cast<double>(generation) / static_cast<double>(this->max_generations));

        calculate_fitness();

        if (this->log_status)
            log_algo_status();

        drop_least_fit();

        crossover(); 

        mutation();
    }

    // Progress larger than one clears bar
    print_progress_bar(2.0);

    calculate_fitness();

    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6;

    BOOST_LOG_SEV(logger, success) << "Algorithm finished in " << duration << " seconds, shortest distance found:\t" << this->population[0].score;

    // If an output file path has been given write the solution, exection time and final score to this file.
    if (!this->output_file_path.empty()){
        std::ofstream output;
        output.open(this->output_file_path);

        if (!output.is_open())
          BOOST_LOG_SEV(logger, error) << "Unable to open output file:\t" << this->output_file_path.string();
        else{
            output << "Solution:\t[ ";
            for (auto s: this->population[0].data){
                output << s << " "; 
            }
            output << "]" << std::endl;
            output << "Distance:\t" << this->population[0].score << std::endl;
            output << "Duration:\t" << duration << std::endl;
        }
        output.close();
    }
    
}

