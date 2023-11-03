class Config:
    size : int = 20
    num_good_agents : int = 5
    num_adversarial_agents: int = 2
    obs_radius: int = 20
    iterations: int = 100
    # Possible moves for each drone. Key is the action, value is the (dx, dy) tuple
    # Default: 0 - stay, 1 - up, 2 - down, 3 - left, 4 - right
    possible_moves : {int : int} = {0 : (0, 0), 1 : (0, 1), 2 : (0, -1), 3 : (-1, 0), 4 : (1, 0)}
