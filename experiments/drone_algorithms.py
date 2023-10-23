# drone_algorithms.py

import random
import numpy as np

def greedy_decision(drone, all_drones, drone_index, D, distance_to_origin_weight=1, epsilon=0.1):
    min_value = float('inf')
    best_dir = (0, 0)

    # With probability epsilon, choose a random direction
    if random.random() < epsilon:
        return (random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))

    # Otherwise, use the greedy approach
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            cost = distance_to_origin_weight*(sum([np.sqrt(d.x**2 + d.y**2) for i, d in enumerate(all_drones) if i != drone_index]) + np.sqrt((drone.x + dx)**2 + (drone.y + dy)**2)) / len(all_drones)

            for j, other_drone in enumerate(all_drones):
                if j != drone_index:
                    dist = np.sqrt((drone.x + dx - other_drone.x)**2 + (drone.y + dy - other_drone.y)**2)
                    cost += (dist - D[drone_index][j])**2

            if cost < min_value:
                min_value = cost
                best_dir = (dx, dy)

    return best_dir
