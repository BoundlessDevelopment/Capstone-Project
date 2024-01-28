import numpy as np
import sys
import csv
import os

sys.path.append("..")  # Adds higher directory to python modules path.
from baseline import main as baseline_run
import env.nepiada as nepiada

class SimulationTester:
    def __init__(self, included_data=None, num_runs=1, config_file=None):
        """
        Initialize the SimulationTester with specified data components to include, number of simulation runs,
        and an optional configuration file.
        :param included_data: List of strings indicating which data components to include in results.
        :param num_runs: Number of times to run the simulation.
        :param config_file: Optional configuration file for the simulation.
        """
        self.num_runs = num_runs
        self.config_file = config_file
        if included_data is None:
            self.included_data = ["observations", "rewards", "terminations", "truncations", "infos"]
        else:
            self.included_data = included_data

    def run_simulation(self, config_file=None):
        """
        Run the simulation and store the results.
        :param config_file: Optional configuration file for the simulation.
        """
        if config_file is None:
            config_file = self.config_file

        if config_file:
            # If a config file is provided, pass it to baseline_run
            self.results, self.agents, self.config, self.env = baseline_run(included_data=self.included_data, config_file=config_file)
        else:
            # If no config file is provided, call baseline_run without the config_file parameter
            self.results, self.agents, self.config, self.env = baseline_run(included_data=self.included_data)


    def calculate_convergence_score(self):
        """
        This function calculates the convergence score based on the average reward across all agents 
        in the last iteration of the algorithm.

        IMPORTANT: An ideal / globally optimal NE will have a score of zero. Lower the score the closer 
        it is to globally optimal NE.
        """

        if not hasattr(self, 'results'):
            print("Cannot compute convergence score before a simulation is run. Run the simulation first.")
            return -1

        # Get the cost from the latest reward functions
        net_cost = 0.0
        last_iteration = self.results[self.config.iterations - 1]
        num_agents = 0
        for agent, reward in last_iteration["rewards"].items():
            net_cost += reward
            num_agents += 1

        convergence_score = -1 * (net_cost / num_agents)
        assert convergence_score >= 0, "Convergence score cannot be negative"

        return convergence_score

    def run_multiple_simulations(self):
        """
        Run the simulation multiple times and calculate the average convergence score.
        """
        scores = []
        for _ in range(self.num_runs):
            self.run_simulation(self.config_file)
            score = self.calculate_convergence_score()
            if score != -1:
                scores.append(score)

        return np.mean(scores) if scores else -1

    def print_results(self):
        """
        Print the stored simulation results and write them to a CSV file in a specified directory.
        """
        if not hasattr(self, 'results'):
            print("No results to display. Run the simulation first.")
            return

        # Ensure the simulation directory exists
        if not os.path.exists(self.config.simulation_dir):
            os.makedirs(self.config.simulation_dir)

        csv_filename = os.path.join(self.config.simulation_dir, "simulation_results.csv")

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Writing header
            headers = ['Step'] + list(self.results[0].keys())
            writer.writerow(headers)

            # Writing data and printing to console
            for step, step_result in enumerate(self.results):
                print(step_result)  # Printing to console
                row = [step] + list(step_result.values())
                writer.writerow(row)

            convergence_score = self.calculate_convergence_score()
            print("\nCalculating score, ideal NE is (0): ", convergence_score)
            writer.writerow(["Convergence Score", convergence_score])

        print(f"Results written to {csv_filename}")

# Example usage
if __name__ == "__main__":
    tester = SimulationTester(num_runs=1, config_file="test_cases/default.json")
    average_score = tester.run_multiple_simulations()
    print("Average Convergence Score over", tester.num_runs, "runs:", average_score)
    tester.print_results()
