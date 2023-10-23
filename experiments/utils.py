import numpy as np
import matplotlib.pyplot as plt

def compute_adjacency_matrix(all_drones):
    n = len(all_drones)
    matrix = [[0] * n for _ in range(n)]
    for i, drone in enumerate(all_drones):
        visible = drone.visible_drones(all_drones)
        for v in visible:
            j = all_drones.index(v)
            matrix[i][j] = 1
    return matrix

def worker(drone, all_drones, idx, D, distance_to_origin_weight, epsilon):
    drone.calculate_direction(all_drones, idx, D, distance_to_origin_weight=distance_to_origin_weight, epsilon=epsilon)
    drone.move()  # move the drone after calculation
    return drone

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

def plot_drone_movements(positions_history):
    plt.figure(figsize=(5,5))
    for iteration, positions in enumerate(positions_history):
        for idx, (x, y) in enumerate(positions):
            color = ['red', 'blue', 'green', 'orange', 'purple', 'brown'][idx % 6]
            plt.plot(x, y, color, label="Drone" + str(idx) if iteration == 1 else "", alpha=0.1)
            plt.text(x, y, str(iteration), color=color, fontsize=8, ha='center', va='center')

    plt.title('Drone Movements Over Iterations')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    plt.show()
