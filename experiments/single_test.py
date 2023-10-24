from drone_simulation import simulate_environment
from drone_algorithms import greedy_decision, mean_update, median_update

import numpy as np

def test_simulation():
    N = 4
    D = [
        [0, 7.07, 10, 7.07], 
        [7.07, 0, 7.07, 10], 
        [10, 7.07, 0, 7.07], 
        [7.07, 10, 7.07, 0]
    ]

    # Generating random starting positions for the drones within a 20x20 square
    random_positions = [(np.random.uniform(0, 20), np.random.uniform(0, 20)) for _ in range(N)]
    avg_score, iteration = simulate_environment(N, D, decision_function=greedy_decision, belief_update_function=mean_update, observation_radius=0, communication_radius=1000, num_adversarial=0) 

    print(f"Average Score: {avg_score}, Iteration Taken: {iteration}")

if __name__ == "__main__":
    test_simulation()
