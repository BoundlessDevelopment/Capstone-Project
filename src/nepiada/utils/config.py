# TODO: Make an enum
# Default: 0 - stay, 1 - up, 2 - down, 3 - left, 4 - right

class Config:
    """ 
    We allow our environment to be customizable based on the user's requirements.
    The a brief description of all parameters is given below

    dim: Dimension of the environment
    size: The size of each dimension of the environment
    iterations: Max number of iterations allowed
    num_good_agents: Number of truthful agents
    num_adversarial_agents: Number of rogue or un-truthful agents

    dynamic_obs: Set to true if you wish to update observation graph based on agent's current position
    obs_radius: The only supported form of dynamic observation is including proximal agents that fall within the observation radius
    communication_all: If set to True all agents should be able to communicate with all other agents

    possible_moves: The valid actions for each agent
    """
    # Initialization parameters
    dim : int = 2
    size : int = 20
    iterations : int = 100
    num_good_agents : int = 5
    num_adversarial_agents: int = 2

    # Graph update parameters
    dynamic_obs: bool = True
    obs_radius: int = 5
    full_communication: bool = True

    # Agent update parameters 
    # Possible moves for each drone. Key is the action, value is the (dx, dy) tuple
    possible_moves : {int : int} = {0 : (0, 0), 1 : (0, 1), 2 : (0, -1), 3 : (-1, 0), 4 : (1, 0)}
    empty_cell : int = -1


