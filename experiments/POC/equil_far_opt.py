import sys
sys.path.append("..")  # Adds higher directory to python modules path

from drone_simulation import simulate_environment

import numpy as np

def test_simulation():
    N = 3
    D = [[0, 10, 10], 
         [10, 0, 10], 
         [10, 10, 0]]

    
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

    avg_score = simulate_environment(N, D, initial_positions=optimal_positions, distance_to_origin_weight=10, epsilon=0.0, plot_results=True)
    print(f"Average Score: {avg_score}")

if __name__ == "__main__":
    test_simulation()

    
    