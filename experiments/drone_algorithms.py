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

def mean_update(drone, all_drones, all_share_vectors, drone_index):
    visible_drones_set = drone.visible_drones(all_drones)
    communicating_drones_set = drone.communicating_drones(all_drones)

    for idx, other_drone in enumerate(all_drones):
        # If the drone is directly visible, update its location directly
        if other_drone in visible_drones_set:
            drone.beliefs[idx] = (other_drone.x, other_drone.y)
        else:
            # If the drone is not directly visible, then update its belief based on the communicating drones' shared vectors
            other_beliefs = [b[idx] for i, b in enumerate(all_share_vectors) if all_drones[i] in communicating_drones_set and b[idx] != (None, None)]
            if other_beliefs:
                avg_x = sum(x for x, _ in other_beliefs) / len(other_beliefs)
                avg_y = sum(y for _, y in other_beliefs) / len(other_beliefs)
                drone.beliefs[idx] = (avg_x, avg_y)

def median_update(drone, all_drones, all_share_vectors, drone_index):
    visible_drones_set = drone.visible_drones(all_drones)
    communicating_drones_set = drone.communicating_drones(all_drones)

    for idx, other_drone in enumerate(all_drones):
        # If the drone is directly visible, update its location directly
        if other_drone in visible_drones_set:
            drone.beliefs[idx] = (other_drone.x, other_drone.y)
        else:
            # If the drone is not directly visible, then update its belief based on the communicating drones' shared vectors
            other_beliefs = [b[idx] for i, b in enumerate(all_share_vectors) if all_drones[i] in communicating_drones_set and b[idx] != (None, None)]
            if other_beliefs:
                xs = [x for x, _ in other_beliefs]
                ys = [y for _, y in other_beliefs]
                
                # Compute median of x and y coordinates
                median_x = np.median(xs)
                median_y = np.median(ys)
                drone.beliefs[idx] = (median_x, median_y)
