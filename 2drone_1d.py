import random
import matplotlib.pyplot as plt

class Drone:
    def __init__(self, position=0):
        self.position = position
        self.intended_direction = 0  # -1: left, 0: stay, 1: right

    def calculate_direction(self, other_position, D, other_intended_direction=0):
        # Objective values with the knowledge of the other drone's intended direction
        current_value = self.position**2 + (abs(self.position - (other_position + other_intended_direction)) - D)**2

        # Simulate moving left
        left_value = (self.position-1)**2 + (abs(self.position-1 - (other_position + other_intended_direction)) - D)**2

        # Simulate moving right
        right_value = (self.position+1)**2 + (abs(self.position+1 - (other_position + other_intended_direction)) - D)**2

        # Choose the direction that minimizes the objective value
        if left_value < current_value and left_value <= right_value:
            self.intended_direction = -1
        elif right_value < current_value and right_value <= left_value:
            self.intended_direction = 1
        else:
            self.intended_direction = 0

    def move(self):
        self.position += self.intended_direction

def simulate_environment(D):
    # Initialize drones at random positions
    drone1 = Drone(random.randint(-10, 10))
    drone2 = Drone(random.randint(-10, 10))

    iteration = 0
    plt.figure(figsize=(10, 5))

    while True:
        iteration += 1
        print(f"Iteration {iteration}: Drone1: {drone1.position}, Drone2: {drone2.position}")

        # Plotting
        plt.plot([drone1.position], [iteration], 'ro', label='Drone1' if iteration == 1 else "")
        plt.plot([drone2.position], [iteration], 'bo', label='Drone2' if iteration == 1 else "")

        # Initial decision
        drone1.calculate_direction(drone2.position, D)
        drone2.calculate_direction(drone1.position, D)

        # Recalculate direction after sharing intended directions
        drone1.calculate_direction(drone2.position, D, drone2.intended_direction)
        drone2.calculate_direction(drone1.position, D, drone1.intended_direction)

        # If neither drone wants to move, break out of loop
        if drone1.intended_direction == 0 and drone2.intended_direction == 0:
            break

        # Drones move
        drone1.move()
        drone2.move()

    # Configure and show plot
    plt.title('Drone Movements Over Iterations')
    plt.xlabel('Position on Number Line')
    plt.ylabel('Iteration')
    plt.legend()
    plt.grid(True)
    plt.show()

    return drone1.position, drone2.position

D = 5  # You can modify this value as needed
final_positions = simulate_environment(D)
print("\nFinal positions of drone1 and drone2:", final_positions)
