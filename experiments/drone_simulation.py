from drone_algorithms import greedy_decision
from utils import compute_distance, worker, individual_drone_score, plot_drone_movements, compute_share_vector
import random
import concurrent.futures
import numpy as np

class Drone:
    def __init__(self, decision_function, position=(0.0,0.0), observation_radius=50, communication_radius=50, num_drones=0, adversarial=False):
        self.x, self.y = position
        self.intended_direction = (0, 0)
        self.decision_function = decision_function
        self.current_cost = float('inf')
        self.observation_radius = observation_radius
        self.communication_radius = communication_radius
        self.beliefs = [(None, None) for _ in range(num_drones)]
        self.adversarial = adversarial

    def visible_drones(self, all_drones):
        """ Return a list of drones that are visible to the current drone. """
        return [d for d in all_drones if compute_distance((self.x, self.y), (d.x, d.y)) <= self.observation_radius]

    def communicating_drones(self, all_drones):
        """ Return a list of drones that can communicate with the current drone. """
        return [d for d in all_drones if compute_distance((self.x, self.y), (d.x, d.y)) <= self.communication_radius]
    
    def get_share_vector(self):
        """ Return the information this drone shares with others. """
        if self.adversarial:
            noise = [(np.random.normal(0, 1), np.random.normal(0, 1)) for _ in self.beliefs]
            return [(belief[0] + noise[i][0] if belief[0] is not None else None,
                     belief[1] + noise[i][1] if belief[1] is not None else None) for i, belief in enumerate(self.beliefs)]
        else:
            return self.beliefs.copy()
    
    def update_beliefs(self, all_drones, all_share_vectors, drone_index):
        visible_drones_set = self.visible_drones(all_drones)
        communicating_drones_set = self.communicating_drones(all_drones)

        for idx, drone in enumerate(all_drones):
            # If the drone is directly visible, update its location directly
            if drone in visible_drones_set:
                self.beliefs[idx] = (drone.x, drone.y)
            else:
                # If the drone is not directly visible, then update its belief based on the communicating drones' shared vectors
                other_beliefs = [b[idx] for i, b in enumerate(all_share_vectors) if all_drones[i] in communicating_drones_set and b[idx] != (None, None)]
                if other_beliefs:
                    avg_x = sum(x for x, _ in other_beliefs) / len(other_beliefs)
                    avg_y = sum(y for _, y in other_beliefs) / len(other_beliefs)
                    self.beliefs[idx] = (avg_x, avg_y)

    def calculate_direction(self, all_drones, drone_index, D, **kwargs):
        beliefs_positions = [pos for pos in self.beliefs if pos != (None, None)]
        self.intended_direction = self.decision_function(self, beliefs_positions, drone_index, D, **kwargs)

    def move(self):
        self.x += self.intended_direction[0]
        self.y += self.intended_direction[1]

def simulate_environment(N, D, initial_positions=None, decision_function=greedy_decision, distance_to_origin_weight=5, epsilon=0.1, verbose=True, adversarial_indices=[], observation_radius=50, communication_radius=50):
    
    if initial_positions is None:
        drones = [Drone(decision_function, (random.randint(-50,50), random.randint(-50,50)), observation_radius=observation_radius, communication_radius=communication_radius, num_drones=N, adversarial=(i in adversarial_indices)) for i in range(N)]
    else:
        drones = [Drone(decision_function, pos, observation_radius=observation_radius, communication_radius=communication_radius, num_drones=N, adversarial=(i in adversarial_indices)) for i, pos in enumerate(initial_positions)]

    iteration = 0
    positions_history = []

    while True:
        iteration += 1
        current_positions = [(drone.x, drone.y) for drone in drones]
        positions_history.append(current_positions)

        # Parallelized computation of share vectors
        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_share_vectors = list(executor.map(compute_share_vector, drones))

        if verbose:
            print(f"Iteration {iteration}: {current_positions}")
            for idx, drone in enumerate(drones):
                print(f"Cost for Drone {idx}: {individual_drone_score(drone, drones, idx, D)}")

        if positions_history.count(current_positions) > 2:
            break

        with concurrent.futures.ThreadPoolExecutor() as executor:
            drones = list(executor.map(worker, drones, [drones]*N, [all_share_vectors]*N, range(N), [D]*N, [distance_to_origin_weight]*N, [epsilon]*N))

    if verbose:
        plot_drone_movements(positions_history)

    avg_score = sum([individual_drone_score(drone, drones, idx, D) for idx, drone in enumerate(drones)]) / N
    return avg_score
