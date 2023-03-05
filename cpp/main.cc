#include<iostream>
#include<fstream>
#include<filesystem>
#include<tuple>
#include<vector>

#include<boost/log/core.hpp>
#include<boost/log/trivial.hpp>
#include<boost/log/utility/setup/common_attributes.hpp>
#include<boost/log/utility/setup/console.hpp>
#include<boost/log/expressions.hpp>
#include<boost/log/sinks/text_file_backend.hpp>
#include<boost/log/utility/setup/file.hpp>
#include<boost/program_options.hpp>

#include "ts_genetic_solver.h"

namespace po = boost::program_options;
namespace logging = boost::log;
namespace keywords = boost::log::keywords;

#define DEFAULT_N_GENERATIONS (100)
#define DEFAULT_DROP_FRACTION (0.2)
#define DEFAULT_MUTATION_FRACTION (0.1)
#define DEFAULT_POP_SIZE (100)

po::variables_map parse_args(int argc, char* argv[])
{
    po::options_description desc("Genetic Solver for Traveling Salesman Problem");
    desc.add_options()
        ("help", "Produce help message")
        ("graph", po::value<std::string>(), "Path to graph file.")
        ("generations", po::value<unsigned>()->default_value(DEFAULT_N_GENERATIONS), "The number of generations to run the algorithm.")
        ("pop-size", po::value<unsigned long>()->default_value(DEFAULT_POP_SIZE), "The size of the chromosome population")
        ("drop-frac", po::value<double>()->default_value(DEFAULT_DROP_FRACTION), "Fraction of chromosomes to be dropped every generation.")
        ("mutation-frac", po::value<double>()->default_value(DEFAULT_MUTATION_FRACTION), "Fraction of chromosome to be mutated every generation.")
        ("seed", po::value<unsigned>()->default_value(42), "Seed to use for random number generation")
        ("no-print", po::bool_switch()->default_value(false), "Disable printing to console.")
        ("output", po::value<std::string>(), "Path to file where results will be stored. Optional.")
    ;

    // Add positional options
    po::positional_options_description p_desc; 
    p_desc.add("graph", 1);

    po::variables_map vm;
    try {
        // Do parsing
        po::store(po::command_line_parser(argc, argv).options(desc).positional(p_desc).run(), vm);
        po::notify(vm);
    } catch (std::exception const &e) {
      std::cerr << e.what() << std::endl;
      std::cout << "\n" << desc << std::endl;
    }

    if (vm.count("help"))
        std::cout << "\n" << desc << std::endl;
    if (vm.count("graph") == 0)
        std::cout << "Error: No graph supplied.\n" << desc << std::endl;

    return vm;
}

std::vector<std::vector<long>>* read_graph_from_file(std::filesystem::path path)
{
    std::vector<std::vector<long>>* graph;
    unsigned long num_nodes; 
    unsigned short ix, iy;
    std::string buffer;
    std::ifstream in;    

    // Check if file exists
    if (!std::filesystem::exists(path)) {
        std::cerr << "Graph file not found:\t" << path.string() << std::endl;
        return nullptr;
    }

    if (std::filesystem::is_directory(path)){
        std::cout << "Graph file is directory:\t" << path.string() << std::endl;
        return nullptr;
    }

    // Open file
    in.open(path, std::ios::in);

    if (in.fail()) {
        std::cerr << "Unable to open file:\t" << path.string() << std::endl;
        return nullptr;
    } 

    // Initialize graph
    graph = new std::vector<std::vector<long>>;

    std::getline(in, buffer); 
    num_nodes = std::stoi(buffer);
    
    // Read line by line the graph matrix from file
    ix = 0;
    iy = 0;

    std::vector<long> temp_vec;
    while (!in.eof()) {

        in >> buffer;
        temp_vec.push_back(std::stoi(buffer));
        ix++;

        if (ix == num_nodes) {
            (*graph).push_back(temp_vec);
            ix = 0;
            iy++;
            temp_vec.clear();
        }

    }

    return graph;
}


int main(int argc, char* argv[]){
  
    auto args = parse_args(argc, argv);

    if (args.count("help"))
        return EXIT_SUCCESS;
    if (args.count("graph") == 0)
        return EXIT_FAILURE;

    // Setup logging
    logging::add_common_attributes();
    logging::add_file_log(keywords::file_name = "logs/%Y-%m-%d_%H-%M-%s_cpp.log", keywords::format = "[%TimeStamp%]: %Message%");

    if (!args["no-print"].as<bool>())
        logging::add_console_log(std::cout, keywords::format = ">>> %Message%");

    std::vector<std::vector<long>> *graph;

    // Use resolved path of the supplied graph file
    auto graph_path = std::filesystem::absolute(std::filesystem::path(args["graph"].as<std::string>()));

    // Read file, if it fails abort program
    if (!(graph = read_graph_from_file(graph_path)))
        return EXIT_FAILURE;

    BOOST_LOG_TRIVIAL(info) << "Graph loaded from file:\t" << graph_path.string();

    // Create and load argument structure
    auto solver_args = TravelingSalesmanGeneticSolverArgs();
    solver_args.max_generations = args["generations"].as<unsigned>();
    solver_args.drop_frac = args["drop-frac"].as<double>();
    solver_args.mutation_frac = args["mutation-frac"].as<double>();
    solver_args.seed = args["seed"].as<unsigned>();
    solver_args.pop_size = args["pop-size"].as<unsigned long>();
    
    // If output file arg is given send it to the solver otherwise send empty path
    if (args.count("output"))
        solver_args.output = std::filesystem::absolute(std::filesystem::path(args["output"].as<std::string>()));
    else
        solver_args.output = std::filesystem::path();

    TravelingSalesmanGeneticSolver solver(*graph, solver_args);

    // When graph has been loaded into memory in the solver free it here
    delete graph;

    // Run solver
    solver.run();

    return EXIT_SUCCESS;
}
