import sys
sys.path.insert(0, "..")  # Insert at the beginning of sys.path

from drone_algorithms import greedy_decision
from drone_simulation import Drone, simulate_environment
from utils import compute_distance, worker, individual_drone_score, plot_drone_movements
import random
import concurrent.futures
import numpy as np


def test_drone_initialization():
    drone = Drone(greedy_decision)
    assert drone.x == 0.0 and drone.y == 0.0, f"Expected (0.0, 0.0) but got ({drone.x}, {drone.y})"


def test_visible_drones():
    drone1 = Drone(greedy_decision, (0, 0))
    drone2 = Drone(greedy_decision, (25, 25))
    drone3 = Drone(greedy_decision, (60, 60))
    drones = [drone1, drone2, drone3]
    
    visible = drone1.visible_drones(drones)
    assert drone2 in visible and drone3 not in visible, "Visibility detection failed!"


def test_communicating_drones():
    drone1 = Drone(greedy_decision, (0, 0))
    drone2 = Drone(greedy_decision, (25, 25))
    drone3 = Drone(greedy_decision, (60, 60))
    drones = [drone1, drone2, drone3]
    
    communicating = drone1.communicating_drones(drones)
    assert drone2 in communicating and drone3 not in communicating, "Communication detection failed!"


def test_simulation_termination():
    N = 3
    D = [
        [0, 10, 10], 
        [10, 0, 10], 
        [10, 10, 0]
    ] # Assuming some value for D
    score = simulate_environment(N, D, verbose=False)
    assert isinstance(score, float), "Simulation did not return the average score correctly!"


def test_belief_update():
    drone1 = Drone(greedy_decision, (0, 0), num_drones=2)
    drone2 = Drone(greedy_decision, (25, 25), num_drones=2)
    drones = [drone1, drone2]
    beliefs = [drone.beliefs for drone in drones]

    drone1.update_beliefs(drones, beliefs, 0)
    assert drone1.beliefs[1] == (25, 25), f"Expected beliefs to be updated but got {drone1.beliefs[1]}"

