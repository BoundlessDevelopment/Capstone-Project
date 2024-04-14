#!/usr/bin/env "C:/Users/Hetav Pandya/AppData/Local/Programs/Python/Python311/python.exe"

# Define the different types of noise and observation radius
number_of_runs=10
noise_types=("gaussian" "uniform" "laplacian" "randomize")
observation_radii=(1 2 4 8 16)
iterations=(10 20 50 70)
logs=0

# Delete a file named convergence_metrics.txt if it exists
if [ -f convergence_metrics.txt ]; then
    rm convergence_metrics.txt
fi

# Get all the requirements
# pip3 install -r requirements.txt
python3 -m pip install numpy
python3 -m pip install gym
python3 --version

echo $PATH

# Iterate for number_of_runs times
for seed in $(seq 1 $number_of_runs); do
    #Iterate over different number of iterations
    for iteration in "${iterations[@]}"; do
        # Iterate over each noise type
        for noise in "${noise_types[@]}"; do
            # Iterate over each observation radius
            for radius in "${observation_radii[@]}"; do
                # Run the baseline.py script and save the output
                python3 baseline.py $seed 2 4 3 2 $radius $noise $iteration > baseline_logs_${seed}_${iteration}_${noise}_${radius}.txt

                # Select the last line from output.txt and save it in a variable
                score=$(tail -n 1 baseline_logs_${seed}_${iteration}_${noise}_${radius}.txt)

                # Append this score to a file named convergence_metrics.txt
                echo "baseline $seed 2 4 3 2 $iteration $noise $radius $score" >> convergence_metrics.txt

                # Delete the baseline.txt file if logs is set to 0
                if [ $logs -eq 0 ]; then
                    rm baseline_logs_${seed}_${iteration}_${noise}_${radius}.txt
                fi

                # Copy all images from the plot folder to a new folder named baseline_{noise}_{observation_radius}
                mkdir -p plots/baseline_${seed}_${iteration}_${noise}_${radius}
                cp plots/*.png plots/baseline_${seed}_${iteration}_${noise}_${radius}

                # # Run the rl_ma_dqn_experimental.py script
                # python3 rl_ma_dqn_experimental.py $seed 2 4 3 2 $radius $noise $iteration > rl_ma_dqn_experimental_logs_${seed}_${iteration}_${noise}_${radius}.txt

                # # Select the last line from output.txt and save it in a variable
                # score = $(tail -n 1 rl_ma_dqn_experimental_logs_${seed}_${iteration}_${noise}_${radius}.txt)

                # # Append this score to a file named convergence_metrics.txt
                # echo "rl_ma_dqn_experimental $seed 2 4 3 2 $iteration $noise $radius $score" >> convergence_metrics.txt

                # # Delete the baseline.txt file if logs is set to 0
                # if [ $logs -eq 0 ]; then
                #     rm rl_ma_dqn_experimental_logs_${seed}_${iteration}_${noise}_${radius}.txt
                # fi

                # # Copy all images from the plot folder to a new folder named rl_ma_dqn_experimental_{noise}_{observation_radius}
                # mkdir -p plots/rl_ma_dqn_experimental_${seed}_${iteration}_${noise}_${radius}
                # cp plots/*.png plots/rl_ma_dqn_experimental_${seed}_${iteration}_${noise}_${radius}
            done
        done
    done
done