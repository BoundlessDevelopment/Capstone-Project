from drone_simulation import simulate_environment
from drone_algorithms import greedy_decision

import numpy as np

def test_simulation():
    N = 3
    D = [
        [0, 10, 10], 
        [10, 0, 10], 
        [10, 10, 0]
    ]

    # Generating random starting positions for the drones within a 20x20 square
    random_positions = [(np.random.uniform(0, 20), np.random.uniform(0, 20)) for _ in range(N)]
    avg_score, iteration = simulate_environment(N, D, observation_radius=0, communication_radius=1000, num_adversarial=1)  # Sets drones at indices 2 and 4 as adversarial



    print(f"Average Score: {avg_score}, Iteration Taken: {iteration}")

if __name__ == "__main__":
    test_simulation()
