class Config:
    size : int = 20
    num_good_agents : int = 5
    num_adversarial_agents: int = 2
    obs_radius: int = 20
    iterations: int = 100
    possible_moves : {int : int} = {0 : (0, 0), 1 : (0, 1), 2 : (0, -1), 3 : (-1, 0), 4 : (1, 0)}
