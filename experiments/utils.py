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
from matplotlib.animation import FuncAnimation

def update(frame, positions_history, scatters, texts, adversarial_flags):
    for idx, (x, y) in enumerate(positions_history[frame]):
        scatters[idx].set_offsets([x, y])
        texts[idx].set_position((x, y))
        label_text = "Adv" if adversarial_flags[idx] else str(frame)  # Label adversarial drones with "Adv"
        texts[idx].set_text(label_text)

def plot_drone_movements(positions_history, adversarial_flags, save_gif=True, filename='noadv.gif'):
    fig, ax = plt.subplots(figsize=(5,5))
    scatters = []
    texts = []

    normal_color = 'blue'  # Color for non-adversarial drones
    adv_color = 'red'  # Distinct color for adversarial drones

    for idx, (x, y) in enumerate(positions_history[0]):
        color = adv_color if adversarial_flags[idx] else normal_color
        marker = 'D' if adversarial_flags[idx] else 'o'  # Different shape for adversarial drones
        scatter = ax.scatter(x, y, s=100, c=color, marker=marker, edgecolors='black', alpha=0.7, label=f"Drone {idx} {'(Adv)' if adversarial_flags[idx] else ''}")
        scatters.append(scatter)
        text = ax.text(x, y, "", color=color, fontsize=8, ha='center', va='center')
        texts.append(text)

    max_range = max(max([abs(x) for positions in positions_history for x, _ in positions]),
                    max([abs(y) for positions in positions_history for _, y in positions]))
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    # Marking the origin (0, 0) with an "X"
    ax.text(0, 0, 'X', color='black', fontsize=12, ha='center', va='center')

    ax.set_title('Drone Movements Over Iterations')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.grid(True)
    ax.legend(loc='upper right', fontsize='small')

    anim = FuncAnimation(fig, update, frames=len(positions_history), fargs=(positions_history, scatters, texts, adversarial_flags), interval=200, repeat=False)

    if save_gif:
        anim.save(filename, writer='pillow', fps=5)

    plt.show()

# Example usage:
# adversarial_flags = [False, True, False, True, False]  # Example: Drones 1 and 3 are adversarial
# plot_drone_movements(positions_history, adversarial_flags, save_gif=True, filename='my_drone_animation.gif')







