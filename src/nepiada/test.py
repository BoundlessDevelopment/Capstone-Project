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

    def calculate_cost(self):
        """
        This function calculates the cost value given the agent's beliefs.
        It is based on where each agent is actually located at the end of the simulation.

        It is calculated using a similar function to the cost function in
        Diane and Prof. Pavel's Paper, however modified to the discrete space.
        """

        # Get the cost from the latest reward functions
        net_cost = 0
        last_iteration = self.results[self.config.iterations - 1]
        for agent, reward in last_iteration["rewards"].items():
            net_cost += reward

        return net_cost

    def print_results(self):
        """
        Print the stored simulation results.
        """
        if not hasattr(self, 'results'):
            print("No results to display. Run the simulation first.")
            return

        for step_result in self.results:
            print(step_result)

        print("\nCalculating net cost, ideal NE is (0): ", self.calculate_cost())

# Example usage
if __name__ == "__main__":
    tester = SimulationTester()
    tester.run_simulation()
    tester.print_results()
