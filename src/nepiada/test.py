# Import the main function from the simulation script
from baseline import main
import numpy as np
import env.nepiada as nepiada

class SimulationTester:
    def __init__(self, included_data=None):
        """
        Initialize the SimulationTester with specified data components to include.
        :param included_data: List of strings indicating which data components to include in results.
        """
        if included_data is None:
            self.included_data = ["observations", "rewards", "terminations", "truncations", "infos"]
        else:
            self.included_data = included_data

    def run_simulation(self):
        """
        Run the simulation and store the results.
        """
        self.results, self.agents, self.config, self.env = main(included_data=self.included_data)

    def calculate_convergence_score(self):
        """
        This function calculates the convergence score based on the average reward acrooss all agent 
        in the last iteration of the algorithm.

        The reward of the agents in turn are calculated using two metrics: global arrangement 
        and local arrangement costs, which are described in Pavel and Dian's paper.

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

        return -1 * (net_cost / num_agents)

    def print_results(self):
        """
        Print the stored simulation results.
        """
        if not hasattr(self, 'results'):
            print("No results to display. Run the simulation first.")
            return

        for step_result in self.results:
            print(step_result)

        print("\nCalculating score, ideal NE is (0): ", self.calculate_convergence_score())

# Example usage
if __name__ == "__main__":
    tester = SimulationTester()
    tester.run_simulation()
    tester.print_results()
