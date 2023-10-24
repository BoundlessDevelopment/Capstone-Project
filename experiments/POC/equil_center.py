import sys
sys.path.insert(0, "..")  # Insert at the beginning of sys.path

from drone_simulation import simulate_environment
from drone_algorithms import greedy_decision


import numpy as np

def test_simulation():
    N = 3
    D = [[0, 10, 10], 
         [10, 0, 10], 
         [10, 10, 0]]

    
    optimal_positions = [
        (0, 5*np.sqrt(3)),            # Top vertex of the triangle
        (-5, -2.5*np.sqrt(3)),       # Bottom-left vertex of the triangle
        (5, -2.5*np.sqrt(3))         # Bottom-right vertex of the triangle
    ]

    avg_score = simulate_environment(N, D, 
                                     initial_positions=optimal_positions, 
                                     decision_function=greedy_decision,   # Passing the decision function as a parameter
                                     distance_to_origin_weight=1, 
                                     epsilon=0, 
                                     verbose=True)
    print(f"Average Score: {avg_score}")

if __name__ == "__main__":
    test_simulation()

    
    