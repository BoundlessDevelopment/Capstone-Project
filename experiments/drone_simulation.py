from drone_algorithms import greedy_decision
from utils import compute_adjacency_matrix, worker, individual_drone_score, plot_drone_movements
import random
import concurrent.futures
import numpy as np

class Drone:
    def __init__(self, decision_function, position=(0.0,0.0), observation_radius=50):  # default observation radius
        self.x, self.y = position
        self.intended_direction = (0, 0)
        self.decision_function = decision_function
        self.current_cost = float('inf')
        self.observation_radius = observation_radius

    def visible_drones(self, all_drones):
        """ Return a list of drones that are visible to the current drone. """
        return [d for d in all_drones if np.sqrt((self.x - d.x)**2 + (self.y - d.y)**2) <= self.observation_radius]

    def calculate_direction(self, all_drones, drone_index, D, **kwargs):
        adjacency_matrix = compute_adjacency_matrix(all_drones)
        visible_drones = [all_drones[j] for j in range(len(all_drones)) if adjacency_matrix[drone_index][j] == 1]
        self.intended_direction = self.decision_function(self, visible_drones, drone_index, D, **kwargs)

    def move(self):
        self.x += self.intended_direction[0]
        self.y += self.intended_direction[1]

def simulate_environment(N, D, initial_positions=None, decision_function=greedy_decision, distance_to_origin_weight=1, epsilon=0.1, verbose=True):
    global DISTANCE_TO_ORIGIN_WEIGHT
    DISTANCE_TO_ORIGIN_WEIGHT = distance_to_origin_weight
    
    if initial_positions is None:
        drones = [Drone(decision_function, (random.randint(-50,50), random.randint(-50,50))) for _ in range(N)]
    else:
        drones = [Drone(decision_function, pos) for pos in initial_positions]

    iteration = 0
    positions_history = []

    while True:
        iteration += 1

        # Store positions for detecting repetitive patterns
        current_positions = [(drone.x, drone.y) for drone in drones]
        positions_history.append(current_positions)

        # Verbose mode: Print drone locations and costs
        if verbose:
            print(f"Iteration {iteration}: {current_positions}")
            for idx, drone in enumerate(drones):
                print(f"Cost for Drone {idx}: {individual_drone_score(drone, drones, idx, D)}")

        # Break if the positions repeat
        if positions_history.count(current_positions) > 2:
            break

        with concurrent.futures.ThreadPoolExecutor() as executor:
            drones = list(executor.map(worker, drones, [drones]*N, range(N), [D]*N, [distance_to_origin_weight]*N, [epsilon]*N))

    if verbose:
        plot_drone_movements(positions_history)

    # Calculate average score
    avg_score = sum([individual_drone_score(drone, drones, idx, D) for idx, drone in enumerate(drones)]) / N
    
    return avg_score
