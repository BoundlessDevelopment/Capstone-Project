#!/usr/bin/env python3

import os
import subprocess

# Define the different types of noise and observation radius
number_of_runs = 2
noise_types = ["gaussian", "randomize"]
observation_radii = [2, 16]
iterations = [50]
logs = 0

# Delete a file named convergence_metrics.txt if it exists
if os.path.exists("convergence_metrics.txt"):
    os.remove("convergence_metrics.txt")

# Get all the requirements
# subprocess.run(["python3", "-m", "pip", "install -r", "requirements.txt"])

# Set the directory where the baseline.py script is located
baseline_script_dir = "D:/Hetav_Documents/UofT/Academic_Planning/ECE_496/Capstone-Project/src/nepiada"

# Iterate for number_of_runs times
for seed in range(1, number_of_runs + 1):
    # Iterate over different number of iterations
    for iteration in iterations:
        # Iterate over each noise type
        for noise in noise_types:
            # Iterate over each observation radius
            for radius in observation_radii:
                # Run the baseline.py script and save the output
                subprocess.run(["python", os.path.join(baseline_script_dir, "baseline.py"), str(seed), "2", "4", "3", "2", str(radius), noise, str(iteration)], stdout=open(f"baseline_logs_{seed}_{iteration}_{noise}_{radius}.txt", "w"))

                # Select the last line from output.txt and save it in a variable
                with open(f"baseline_logs_{seed}_{iteration}_{noise}_{radius}.txt", "r") as f:
                    lines = f.readlines()
                    score = lines[-1].strip()

                # Append this score to a file named convergence_metrics.txt
                with open("convergence_metrics.txt", "a") as f:
                    f.write(f"baseline {seed} 2 4 3 2 {iteration} {noise} {radius} {score}\n")

                # Delete the baseline.txt file if logs is set to 0
                if logs == 0:
                    os.remove(f"baseline_logs_{seed}_{iteration}_{noise}_{radius}.txt")

                # Copy all images from the plot folder to a new folder named baseline_{noise}_{observation_radius}
                os.makedirs(f"plots/baseline_{seed}_{iteration}_{noise}_{radius}", exist_ok=True)
                os.system(f"copy \"plots\\all_traj.png\" \"plots\\baseline_{seed}_{iteration}_{noise}_{radius}\"")

                # Delete the png images under plots folder
                os.system("del plots\*.png")

                ### RUN THE DQN SCRIPT ###
                # # Run the baseline.py script and save the output
                # subprocess.run(["python", os.path.join(baseline_script_dir, "src/nepiada/rl_ma_dqn_experimental.py"), str(seed), "2", "4", "3", "2", str(radius), noise, str(iteration)], stdout=open(f"baseline_logs_{seed}_{iteration}_{noise}_{radius}.txt", "w"))

                # # Select the last line from output.txt and save it in a variable
                # with open(f"dqn_logs_{seed}_{iteration}_{noise}_{radius}.txt", "r") as f:
                #     lines = f.readlines()
                #     score = lines[-1].strip()

                # # Append this score to a file named convergence_metrics.txt
                # with open("convergence_metrics.txt", "a") as f:
                #     f.write(f"dqn {seed} 2 4 3 2 {iteration} {noise} {radius} {score}\n")

                # # Delete the baseline.txt file if logs is set to 0
                # if logs == 0:
                #     os.remove(f"dqn_logs_{seed}_{iteration}_{noise}_{radius}.txt")

                # # Copy all images from the plot folder to a new folder named baseline_{noise}_{observation_radius}
                # os.makedirs(f"plots/dqn_{seed}_{iteration}_{noise}_{radius}", exist_ok=True)
                # os.system(f"copy \"plots\\all_traj.png\" \"plots\\dqn_{seed}_{iteration}_{noise}_{radius}\"")

                # # Delete the png images under plots folder
                # os.system("del plots\*.png")
