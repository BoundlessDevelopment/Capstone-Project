
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

    
    x_center, y_center = 50, 50

    x_center, y_center = 50, 50

    # Shifts based on the triangle's centroid
    x_shift = x_center - 5
    y_shift = y_center - (5 * np.sqrt(3) / 3)

    # Updated starting positions as vertices of the translated equilateral triangle
    optimal_positions = [
        (0 + x_shift, 0 + y_shift),                
        (10 + x_shift, 0 + y_shift),               
        (5 + x_shift, 5 * np.sqrt(3) + y_shift)    
    ]

    avg_score = simulate_environment(N, D, 
                                     initial_positions=optimal_positions, 
                                     decision_function=greedy_decision,   # Passing the decision function as a parameter
                                     distance_to_origin_weight=30, 
                                     epsilon=0,
                                     observation_radius= 1000,
                                     communication_radius= 0, 
                                     verbose=True)
    print(f"Average Score: {avg_score}")

if __name__ == "__main__":
    test_simulation()

    
    