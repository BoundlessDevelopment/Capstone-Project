import numpy as np
import matplotlib.pyplot as plt

def compute_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def worker(drone, all_drones, all_beliefs, idx, D, distance_to_origin_weight, epsilon):
    drone.update_beliefs(all_drones, all_beliefs, idx)
    drone.calculate_direction(all_drones, idx, D, distance_to_origin_weight=distance_to_origin_weight, epsilon=epsilon)
    drone.move()  # move the drone after calculation
    return drone

def compute_share_vector(drone):
    return drone.get_share_vector()

def individual_drone_score(drone, all_drones, drone_index, D):
    score = 0
    
    # Drone's distance from the origin
    score += sum([np.sqrt(d.x**2 + d.y**2) for d in all_drones]) / len(all_drones)
    
    # Sum of squared differences from the desired distances
    for j, other_drone in enumerate(all_drones):
        if j != drone_index:
            dist = np.sqrt((drone.x - other_drone.x)**2 + (drone.y - other_drone.y)**2)
            score += (dist - D[drone_index][j])**2
    
    return score

import matplotlib.pyplot as plt

def plot_drone_movements(positions_history):
    plt.figure(figsize=(5,5))
    
    for iteration, positions in enumerate(positions_history):
        for idx, (x, y) in enumerate(positions):
            color = ['red', 'blue', 'green', 'orange', 'purple', 'brown'][idx % 6]
            plt.plot(x, y, color, label="Drone" + str(idx) if iteration == 0 else "", alpha=0.1)
            plt.text(x, y, str(iteration), color=color, fontsize=8, ha='center', va='center')
    
    # Marking the final position of each drone with a big blob
    for idx, (x, y) in enumerate(positions_history[-1]):
        color = ['red', 'blue', 'green', 'orange', 'purple', 'brown'][idx % 6]
        plt.scatter(x, y, s=100, c=color, edgecolors='black', alpha=0.7, label=f"Final Drone {idx}")
    
    plt.title('Drone Movements Over Iterations')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.legend(loc='upper right', fontsize='small')  # Adjust the location and font size here
    plt.grid(True)
    plt.show()

