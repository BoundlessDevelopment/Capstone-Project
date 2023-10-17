import random
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures

class Drone:
    def __init__(self, position=(0,0)):
        self.x, self.y = position
        self.intended_direction = (0, 0)
        self.current_cost = float('inf')  # Initialize with a high cost

    def calculate_direction(self, all_drones, drone_index, D):
        min_value = float('inf')
        best_dir = (0, 0)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cost = (self.x + dx)**2 + (self.y + dy)**2  # Distance from origin
                for j, other_drone in enumerate(all_drones):
                    if j != drone_index:
                        dist = np.sqrt((self.x + dx - other_drone.x)**2 + (self.y + dy - other_drone.y)**2)
                        cost += (dist - D[drone_index][j])**2

                if cost < min_value:
                    min_value = cost
                    best_dir = (dx, dy)

        self.intended_direction = best_dir
        self.current_cost = min_value  # Store the cost of the best direction

    def move(self):
        self.x += self.intended_direction[0]
        self.y += self.intended_direction[1]

def worker(drone, all_drones, idx, D):
    drone.calculate_direction(all_drones, idx, D)
    return drone

def simulate_environment(N, D):
    drones = [Drone((random.randint(-20,20), random.randint(-20,20))) for _ in range(N)]
    iteration = 0

    positions_history = []

    while True:
        iteration += 1
        print(f"Iteration {iteration}: {[(drone.x, drone.y) for drone in drones]}")

        # Print costs of each drone for the current iteration
        for idx, drone in enumerate(drones):
            print(f"Cost for Drone {idx}: {drone.current_cost}")

        # Store positions for detecting repetitive patterns
        current_positions = [(drone.x, drone.y) for drone in drones]
        positions_history.append(current_positions)

        # Break if the positions repeat
        if positions_history.count(current_positions) > 2:
            break

        with concurrent.futures.ThreadPoolExecutor() as executor:
            drones = list(executor.map(worker, drones, [drones]*N, range(N), [D]*N))

        for drone in drones:
            drone.move()

    # Plotting
    plt.figure(figsize=(10,10))
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

N = 3  # Number of drones
D = [[0, 10, 10], 
     [10, 0, 10], 
     [10, 10, 0]]  # Desired distances between drones

simulate_environment(N, D)
