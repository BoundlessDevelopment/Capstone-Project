import sys
sys.path.insert(0, "..")  # Insert at the beginning of sys.path

import pytest
from drone_simulation import Drone, simulate_environment
from drone_algorithms import greedy_decision, mean_update, median_update
from utils import compute_distance

def test_drone_initialization():
    drone = Drone(greedy_decision, mean_update)
    assert drone.x == 0.0 and drone.y == 0.0, f"Expected (0.0, 0.0) but got ({drone.x}, {drone.y})"

def test_visible_drones():
    drone1 = Drone(greedy_decision, mean_update, position=(0, 0), observation_radius=50)
    drone2 = Drone(greedy_decision, mean_update, position=(25, 25), observation_radius=50)
    drone3 = Drone(greedy_decision, mean_update, position=(60, 60), observation_radius=50)
    drones = [drone1, drone2, drone3]
    
    visible = drone1.visible_drones(drones)
    assert drone2 in visible and drone3 not in visible, "Visibility detection failed!"

def test_communicating_drones():
    drone1 = Drone(greedy_decision, mean_update, position=(0, 0), communication_radius=50)
    drone2 = Drone(greedy_decision, mean_update, position=(25, 25), communication_radius=50)
    drone3 = Drone(greedy_decision, mean_update, position=(60, 60), communication_radius=50)
    drones = [drone1, drone2, drone3]
    
    communicating = drone1.communicating_drones(drones)
    assert drone2 in communicating and drone3 not in communicating, "Communication detection failed!"

def test_simulation_termination():
    N = 3
    D = [
        [0, 10, 10], 
        [10, 0, 10], 
        [10, 10, 0]
    ]
    score, iteration = simulate_environment(N, D, verbose=False)
    assert isinstance(score, float), "Simulation did not return the average score correctly!"
    assert isinstance(iteration, int), "Simulation did not return the iteration count correctly!"

def test_belief_update_mean():
    drone1 = Drone(greedy_decision, mean_update, position=(0, 0), num_drones=2, observation_radius=60, communication_radius=60)
    drone2 = Drone(greedy_decision, mean_update, position=(25, 25), num_drones=2, observation_radius=60, communication_radius=60)
    drones = [drone1, drone2]

    drone1.update_beliefs(drones, [drone.get_share_vector() for drone in drones], 0)
    assert drone1.beliefs[1] == (25, 25), f"Expected beliefs to be (25, 25) but got {drone1.beliefs[1]}"

def test_belief_update_median():
    drone1 = Drone(greedy_decision, median_update, position=(0, 0), num_drones=3, observation_radius=60, communication_radius=60)
    drone2 = Drone(greedy_decision, mean_update, position=(25, 25), num_drones=3, observation_radius=60, communication_radius=60)
    drone3 = Drone(greedy_decision, mean_update, position=(50, 50), num_drones=3, observation_radius=60, communication_radius=60)
    drones = [drone1, drone2, drone3]

    # Drone1 has initial beliefs for other drones as (None, None)
    drone1.beliefs = [(0, 0), (None, None), (None, None)]
    drone2.beliefs = [(0, 0), (25, 25), (50, 50)]
    drone3.beliefs = [(0, 0), (25, 25), (50, 50)]

    drone1.update_beliefs(drones, [drone.get_share_vector() for drone in drones], 0)
    # Assuming drone1 updates its belief about drone2 to be the median of [(0, 0), (25, 25)]
    assert drone1.beliefs[1] == (25, 25), f"Expected beliefs to be (25, 25) but got {drone1.beliefs[1]}"

def test_greedy_decision_function():
    drone = Drone(greedy_decision, median_update, position=(0, 0), num_drones=2)
    beliefs_positions = [(0, 0), (25, 25)]

    D = [
        [0, 10],
        [10, 0]
    ]

    drone.calculate_direction(beliefs_positions, 0, D)
    decision = drone.intended_direction

    # This is a basic check. The exact expected decision depends on how the greedy_decision function is implemented.
    # We're assuming that the drone will move in a positive direction since the belief indicates a positive position.
    assert decision[0] >= 0 and decision[1] >= 0, f"Expected a positive direction but got {decision}"


