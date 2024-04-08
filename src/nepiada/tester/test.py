import numpy as np
import sys
import time
from concurrent.futures import ProcessPoolExecutor
sys.path.append("..")  # Adds higher directory to python modules path.

# Import the main function from the simulation script
from baseline import main as baseline_run
import env.nepiada as nepiada

class SimulationTester:
    def __init__(self, included_data=None, k=1):
        """
        Initialize the SimulationTester with specified data components to include and number of simulations to run.
        :param included_data: List of strings indicating which data components to include in results.
        :param k: Number of simulations to run.
        """
        self.k = k
        if included_data is None:
            self.included_data = ["rewards"]
        else:
            self.included_data = included_data
        self.all_results = []
        self.run_times = []

    def single_run(self):
        """
        A single run of the simulation.
        """
        start_time = time.time()
        results, agents, config, env = baseline_run(included_data=self.included_data)
        end_time = time.time()
        return results, end_time - start_time

    def run_simulation(self):
        """
        Run the simulation 'k' times in parallel and store the results and time taken for each run.
        """
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.single_run) for _ in range(self.k)]
            for future in futures:
                results, run_time = future.result()
                self.all_results.append(results)
                self.run_times.append(run_time)

    def calculate_convergence_score(self):
        """
        Calculate the convergence score for each simulation run.
        """
        scores = []
        for results in self.all_results:
            if not results:
                scores.append(-1)
                continue

            net_cost = 0.0
            last_iteration = results[-1]
            num_agents = 0
            for agent, reward in last_iteration["rewards"].items():
                net_cost += reward
                num_agents += 1

            score = -1 * (net_cost / num_agents) if num_agents > 0 else -1
            scores.append(score)

        return scores

    def print_results(self):
        """
        Print the stored simulation results, the convergence scores, and the timing information.
        """
        total_score = 0
        total_time = 0

        scores = self.calculate_convergence_score()
        for i, (score, run_time) in enumerate(zip(scores, self.run_times)):
            print(f"\nConvergence score for Simulation {i+1}: {score}")
            print(f"Time taken for Simulation {i+1}: {run_time} seconds")
            total_score += score
            total_time += run_time

        average_score = total_score / len(scores) if scores else 0
        average_time = total_time / self.k
        print(f"\nAverage Convergence Score over {self.k} runs: {average_score}")
        print(f"Average Time over {self.k} runs: {average_time} seconds")

# Example usage
if __name__ == "__main__":
    start_time = time.time()  # Start timer

    tester = SimulationTester(k=2)  # Initialize SimulationTester to run 2 simulations
    tester.run_simulation()        # Run the simulations
    tester.print_results()         # Print the results

    end_time = time.time()  # End timer
    total_testing_time = end_time - start_time
    print(f"\nTotal Testing Time: {total_testing_time} seconds")
