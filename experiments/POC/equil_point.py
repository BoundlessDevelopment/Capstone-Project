

import sys
sys.path.append("..")  # Adds higher directory to python modules path

from drone_simulation import simulate_environment

import numpy as np

def test_simulation():
    N = 3
    D = [[0, 10, 10], 
         [10, 0, 10], 
         [10, 10, 0]]
   # Updated starting positions as vertices of the translated equilateral triangle
    optimal_positions = [
        (0, 0),                       # Vertex at the origin
        (10, 0),                      # Vertex on the X-axis
        (5, 5 * np.sqrt(3))           # Vertex above the midpoint of the base
    ]

    avg_score = simulate_environment(N, D, initial_positions=optimal_positions, distance_to_origin_weight=1, epsilon=0, verbose=True)
    print(f"Average Score: {avg_score}")

if __name__ == "__main__":
    test_simulation()

    
