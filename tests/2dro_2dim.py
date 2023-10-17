import random
import matplotlib.pyplot as plt

class Drone:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.intended_direction_x = 0  # -1: left, 0: stay, 1: right
        self.intended_direction_y = 0  # -1: down, 0: stay, 1: up

    def calculate_direction(self, other_x, other_y, D, other_intended_direction_x=0, other_intended_direction_y=0):
        # Adjust other's position based on their intended movement
        anticipated_other_x = other_x + other_intended_direction_x
        anticipated_other_y = other_y + other_intended_direction_y

        distance = ((self.x - anticipated_other_x) ** 2 + (self.y - anticipated_other_y) ** 2) ** 0.5
        
        # Objective function calculations for current, left, right, up, and down positions
        current_value = self.x**2 + self.y**2 + (distance - D)**2
        
        left_value = (self.x-1)**2 + self.y**2 + ((self.x-1 - anticipated_other_x)**2 + (self.y - anticipated_other_y)**2 - D**2)
        right_value = (self.x+1)**2 + self.y**2 + ((self.x+1 - anticipated_other_x)**2 + (self.y - anticipated_other_y)**2 - D**2)
        up_value = self.x**2 + (self.y+1)**2 + ((self.x - anticipated_other_x)**2 + (self.y+1 - anticipated_other_y)**2 - D**2)
        down_value = self.x**2 + (self.y-1)**2 + ((self.x - anticipated_other_x)**2 + (self.y-1 - anticipated_other_y)**2 - D**2)

        # Determine best direction based on objective values
        min_value = min(current_value, left_value, right_value, up_value, down_value)
        
        if min_value == left_value:
            self.intended_direction_x, self.intended_direction_y = -1, 0
        elif min_value == right_value:
            self.intended_direction_x, self.intended_direction_y = 1, 0
        elif min_value == up_value:
            self.intended_direction_x, self.intended_direction_y = 0, 1
        elif min_value == down_value:
            self.intended_direction_x, self.intended_direction_y = 0, -1
        else:
            self.intended_direction_x, self.intended_direction_y = 0, 0

    def move(self):
        self.x += self.intended_direction_x
        self.y += self.intended_direction_y

def simulate_environment(D):
    # Initialize drones at random positions
    drone1 = Drone(random.randint(-10, 10), random.randint(-10, 10))
    drone2 = Drone(random.randint(-10, 10), random.randint(-10, 10))
    
    iteration = 0
    plt.figure(figsize=(10, 5))
    
    previous_states = set()
    
    while True:
        iteration += 1
        print(f"Iteration {iteration}: Drone1: ({drone1.x}, {drone1.y}), Drone2: ({drone2.x}, {drone2.y})")
        
        # Plotting
        plt.plot(drone1.x, drone1.y, 'ro', label='Drone1' if iteration == 1 else "")
        plt.plot(drone2.x, drone2.y, 'bo', label='Drone2' if iteration == 1 else "")

        current_state = (drone1.x, drone1.y, drone2.x, drone2.y)
        if current_state in previous_states:
            print("Repetitive movement pattern detected! Breaking out of loop.")
            break
        previous_states.add(current_state)

        # First pass to get initial intended directions
        drone1.calculate_direction(drone2.x, drone2.y, D)
        drone2.calculate_direction(drone1.x, drone1.y, D)

        # Second pass to adjust directions based on the other drone's intentions
        drone1.calculate_direction(drone2.x, drone2.y, D, drone2.intended_direction_x, drone2.intended_direction_y)
        drone2.calculate_direction(drone1.x, drone1.y, D, drone1.intended_direction_x, drone1.intended_direction_y)

        # If neither drone wants to move, break out of loop
        if drone1.intended_direction_x == 0 and drone1.intended_direction_y == 0 and drone2.intended_direction_x == 0 and drone2.intended_direction_y == 0:
            break

        # Drones move
        drone1.move()
        drone2.move()

    # Configure and show plot
    plt.title('Drone Movements Over Iterations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()
    plt.show()

    return (drone1.x, drone1.y), (drone2.x, drone2.y)

D = 5  # You can modify this value as needed
final_positions = simulate_environment(D)
print("\nFinal positions of drone1 and drone2:", final_positions)
