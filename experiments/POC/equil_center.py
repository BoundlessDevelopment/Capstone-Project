import sys
sys.path.insert(0, "..")  # Insert at the beginning of sys.path

from drone_simulation import simulate_environment
from drone_algorithms import greedy_decision

import numpy as np

def test_simulation():
    N = 4
    D = [
        [0, 7.07, 10, 7.07], 
        [7.07, 0, 7.07, 10], 
        [10, 7.07, 0, 7.07], 
        [7.07, 10, 7.07, 0]
    ]
    
    optimal_positions = [
        (0, 5),     # Top vertex
        (-5, 0),    # Left vertex
        (0, -5),    # Bottom vertex
        (5, 0)      # Right vertex
    ]

    avg_score, iteration = simulate_environment(N, D, 
                                     initial_positions=optimal_positions, 
                                     decision_function=greedy_decision,   # Passing the decision function as a parameter
                                     distance_to_origin_weight=1, 
                                     epsilon=0, 
                                     observation_radius= 1000,
                                     communication_radius= 0,
                                     verbose=True)
    print(f"Average Score: {avg_score}, Iteration Taken: {iteration}")

if __name__ == "__main__":
    test_simulation()
