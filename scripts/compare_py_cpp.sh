#!/bin/bash

py_cmd="py/gen_solver.py"
cpp_cmd="./gen_solver"

generations=1000
pop_size=1000
drop_frac=0.25
mutation_frac=0.1

n_execs=10

graph=$1
output=$2

if (( $# != 2 )); then
    echo "Please provied path to graph file and output file as arguments"
    exit 1
fi

check_if_exist () {
    if [[ ! -f "$1" ]]; then
        echo -e "$2:\t$(realpath $1)"
        exit 1
    fi
}

check_if_exist $graph "Graph file not found"
check_if_exist $py_cmd "Python version not found"
check_if_exist $cpp_cmd "C++ version not found"

args="--generations $generations --pop-size $pop_size --drop-frac $drop_frac --mutation-frac $mutation_frac --no-print"

temp_dir=$(mktemp -d)

echo "Comparing python and C++ versions of the TravelingSalesmanGeneticSolver" 

echo -e "Running each algorithm $n_execs times."

#echo -e "Writing temporary results to:\t$temp_dir"

cut_info_from_file () {
    # Read file until it encounters the string in the second argument then print what comes after
    while read -r line
    do
        if [[ $line == "$2"* ]]; then
          echo $line | cut -d : -f2 -
        fi
    done < "$1"
}

py_exec_times=()
cpp_exec_times=()

py_exec_distances=()
cpp_exec_distances=()

# Run the programs n_execs number of time and store the outputs in the temporary folder
for (( idx=1; idx<=$n_execs; idx++ ))
do
    echo -e "Run:\t$idx"

    seed=$RANDOM

    py_output="${temp_dir}/py_out_${idx}"
    cpp_output="${temp_dir}/cpp_out_${idx}" 

    full_py_cmd="$py_cmd $graph $args --output $py_output --seed $seed"
    full_cpp_cmd="$cpp_cmd $graph $args --output $cpp_output --seed $seed"

    eval "$full_py_cmd"
    eval "$full_cpp_cmd"

    py_exec_times+=$(cut_info_from_file $py_output "Duration")
    cpp_exec_times+=$(cut_info_from_file $cpp_output "Duration")

    py_exec_distances+=$(cut_info_from_file $py_output "Distance")
    cpp_exec_distances+=$(cut_info_from_file $cpp_output "Distance")
done

# Python oneliners for calculating mean and standard deviation with floating point precision
py_calc_mean="\"import sys; print('%3.3f' % (sum([float(x) for x in sys.argv[1:]]) / len(sys.argv[1:])))\""
py_calc_std="\"import sys,numpy; print('%3.3f' % numpy.array(sys.argv[1:],dtype=numpy.float64).std())\""

# Calculate statistics about execution time
py_mean_exec=$(eval "python -c $py_calc_mean ${py_exec_times[@]}")
cpp_mean_exec=$(eval "python -c $py_calc_mean ${cpp_exec_times[@]}")

py_std_exec=$(eval "python -c $py_calc_std ${py_exec_times[@]}")
cpp_std_exec=$(eval "python -c $py_calc_std ${cpp_exec_times[@]}")

# Calculate statistics about minimum distance
py_mean_distance=$(eval "python -c $py_calc_mean ${py_exec_distances[@]}")
cpp_mean_distance=$(eval "python -c $py_calc_mean ${cpp_exec_distances[@]}")

py_std_distance=$(eval "python -c $py_calc_std ${py_exec_distances[@]}")
cpp_std_distance=$(eval "python -c $py_calc_std ${cpp_exec_distances[@]}")


echo -e "Comparison of TravelingSalesmanGeneticSolver execution speed of C++ and Python versions." | tee --append $output
echo -e "Graph file used for comparison:\t$(realpath $graph)" | tee --append $output
echo -e "Mean Python version execution time:\t$py_mean_exec ($py_std_exec)" | tee --append $output
echo -e "Mean C++ version execution time:\t$cpp_mean_exec ($cpp_std_exec)" | tee --append $output
echo -e "Mean Python version distance:\t$py_mean_distance ($py_std_distance)" | tee --append $output
echo -e "Mean C++ version distance:\t$cpp_mean_distance ($cpp_std_distance)" | tee --append $output

rm -rf $temp_dir
