# drone_algorithms.py

import random
import numpy as np
from utils import compute_distance

def greedy_decision(drone, beliefs_positions, drone_index, D, distance_to_origin_weight=1, epsilon=0.1):
    min_value = float('inf')
    best_dir = (0, 0)

    # With probability epsilon, choose a random direction
    if random.random() < epsilon:
        return (random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))

    # Otherwise, use the greedy approach
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            new_pos = (drone.x + dx, drone.y + dy)
            
            # Average distance to origin for all believed positions (excluding the current drone)
            avg_distance = sum(compute_distance(pos, (0, 0)) for i, pos in enumerate(beliefs_positions) if i != drone_index) + compute_distance(new_pos, (0, 0))
            avg_distance /= len(beliefs_positions)
            
            cost = distance_to_origin_weight * avg_distance

            for j, other_pos in enumerate(beliefs_positions):
                if j != drone_index:
                    dist = compute_distance(new_pos, other_pos)
                    cost += (dist - D[drone_index][j])**2

            if cost < min_value:
                min_value = cost
                best_dir = (dx, dy)

    return best_dir
