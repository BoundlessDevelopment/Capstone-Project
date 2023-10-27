from drone_simulation import simulate_environment
from drone_algorithms import greedy_decision
import concurrent.futures
import time

def run_simulation(N, D, m, decision_function, num_adversarial):
    weight = 5  # Fixed value for distance_to_origin_weight
    eps = 0.1  # Fixed value for epsilon

    total_score = 0
    total_iterations = 0  # This will hold the total number of iterations for all m simulations
    
    for _ in range(m):
        score, iterations = simulate_environment(N, D, 
                                                 distance_to_origin_weight=weight, 
                                                 epsilon=eps, 
                                                 decision_function=decision_function,
                                                 num_adversarial=num_adversarial,  # Include the number of adversarial agents
                                                 verbose=False)
        total_score += score
        total_iterations += iterations

    average_score = total_score / m
    average_iterations = total_iterations / m
    
    return {
        'distance_to_origin_weight': weight,
        'epsilon': eps,
        'num_adversarial': num_adversarial,  # Include the number of adversarial agents in the result
        'average_score': average_score,
        'average_iterations': average_iterations
    }

def grid_search_simulation(m):
    # Number of adversarial agents
    num_adversarial_agents = [0, 1]

    N = 4
    D = [
        [0, 7.07, 10, 7.07], 
        [7.07, 0, 7.07, 10], 
        [10, 7.07, 0, 7.07], 
        [7.07, 10, 7.07, 0]
    ]

    # Store the results
    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, N, D, m, greedy_decision, num_adv) for num_adv in num_adversarial_agents]

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda x: x['num_adversarial'])

    for result in results:
        print(f"Number of Adversarial Agents: {result['num_adversarial']}, Average Score: {result['average_score']}, Average Iterations: {result['average_iterations']}")

    best_config = min(results, key=lambda x: x['average_score'])
    print("\nBest Configuration:")
    print(f"Number of Adversarial Agents: {best_config['num_adversarial']}, Average Score: {best_config['average_score']}, Average Iterations: {best_config['average_iterations']}")

if __name__ == "__main__":
    start_time = time.time()

    m = 10
    grid_search_simulation(m)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nTotal Time Taken: {elapsed_time:.2f} seconds")
