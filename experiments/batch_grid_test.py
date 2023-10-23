from drone_simulation import simulate_environment
from drone_algorithms import greedy_decision  # Assuming this is where you've saved your decision-making function

import numpy as np
import concurrent.futures
import time  # Importing the time module

def run_simulation(N, D, weight, eps, m, decision_function):
    total_score = 0
    for _ in range(m):
        score = simulate_environment(N, D, 
                                     distance_to_origin_weight=weight, 
                                     epsilon=eps, 
                                     decision_function=decision_function,   # Passing the decision function as a parameter
                                     verbose=False)
        total_score += score
    average_score = total_score / m
    return {
        'distance_to_origin_weight': weight,
        'epsilon': eps,
        'average_score': average_score
    }

def grid_search_simulation(m):
    # Define the grid for distance_to_origin_weight and epsilon
    distance_to_origin_weights = [5]
    epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    N = 3
    D = [
        [0, 10, 10], 
        [10, 0, 10], 
        [10, 10, 0]
    ]

    # Store the results
    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Passing the decision function to each simulation run
        futures = [executor.submit(run_simulation, N, D, weight, eps, m, greedy_decision) for weight in distance_to_origin_weights for eps in epsilons]
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Sort the results by weight and then by epsilon before printing
    results.sort(key=lambda x: (x['distance_to_origin_weight'], x['epsilon']))

    for result in results:
        print(f"Weight: {result['distance_to_origin_weight']}, Epsilon: {result['epsilon']}, Average Score: {result['average_score']}")

    # Return the configuration with the lowest average score
    best_config = min(results, key=lambda x: x['average_score'])
    print("\nBest Configuration:")
    print(f"Weight: {best_config['distance_to_origin_weight']}, Epsilon: {best_config['epsilon']}, Average Score: {best_config['average_score']}")

if __name__ == "__main__":
    start_time = time.time()  # Start the timer

    m = 200  # Number of times each configuration is run
    grid_search_simulation(m)

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time  # Calculate elapsed time

    print(f"\nTotal Time Taken: {elapsed_time:.2f} seconds")  # Print the total time taken
