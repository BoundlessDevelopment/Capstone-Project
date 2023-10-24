from drone_simulation import simulate_environment
from drone_algorithms import greedy_decision
import numpy as np
import concurrent.futures
import time

def run_simulation(N, D, weight, eps, m, decision_function):
    total_score = 0
    total_iterations = 0  # This will hold the total number of iterations for all m simulations
    
    for _ in range(m):
        score, iterations = simulate_environment(N, D, 
                                                 distance_to_origin_weight=weight, 
                                                 epsilon=eps, 
                                                 decision_function=decision_function,
                                                 verbose=False)
        total_score += score
        total_iterations += iterations

    average_score = total_score / m
    average_iterations = total_iterations / m
    
    return {
        'distance_to_origin_weight': weight,
        'epsilon': eps,
        'average_score': average_score,
        'average_iterations': average_iterations
    }

def grid_search_simulation(m):
    # Define the grid for distance_to_origin_weight and epsilon
    distance_to_origin_weights = [5]
    epsilons = [0, 0.1, 0.5]

    N = 3
    D = [
        [0, 10, 10], 
        [10, 0, 10], 
        [10, 10, 0]
    ]

    # Store the results
    results = []


    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, N, D, weight, eps, m, greedy_decision) for weight in distance_to_origin_weights for eps in epsilons]

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: (x['distance_to_origin_weight'], x['epsilon']))

    for result in results:
        print(f"Weight: {result['distance_to_origin_weight']}, Epsilon: {result['epsilon']}, Average Score: {result['average_score']}, Average Iterations: {result['average_iterations']}")

    best_config = min(results, key=lambda x: x['average_score'])
    print("\nBest Configuration:")
    print(f"Weight: {best_config['distance_to_origin_weight']}, Epsilon: {best_config['epsilon']}, Average Score: {best_config['average_score']}, Average Iterations: {best_config['average_iterations']}")

if __name__ == "__main__":
    start_time = time.time()

    m = 50
    grid_search_simulation(m)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nTotal Time Taken: {elapsed_time:.2f} seconds")