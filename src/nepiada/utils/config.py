# TODO: Make an enum
# Default: 0 - stay, 1 - up, 2 - down, 3 - left, 4 - right

from utils.noise import GaussianNoise
from .anim_consts import WIDTH, HEIGHT

class Config:
    """
    We allow our environment to be customizable based on the user's requirements.
    The a brief description of all parameters is given below

    dim: Dimension of the environment
    size: The size of each dimension of the environment
    iterations: Max number of iterations allowed

    agent_grid_width: Width of the final agent formation, note the goal is to have a rectangular formation
    agent_grid_height: Height of the final agent formation, note the goal is to have a rectangular formation
    num_good_agents: Number of truthful agents
    num_adversarial_agents: Number of rogue or un-truthful agents

    dynamic_obs: Set to true if you wish to update observation graph based on agent's current position
    obs_radius: The only supported form of dynamic observation is including proximal agents that fall within the observation radius
    communication_all: If set to True all agents should be able to communicate with all other agents

    possible_moves: The valid actions for each agent
    """

    # Initialization parameters
    dim: int = 2
    size: int = 50
    iterations: int = 5
    simulation_dir: str = "plots"

    # Agent related parameterss
    agent_grid_width: int = 3
    agent_grid_height: int = 3
    num_good_agents: int = 7
    num_adversarial_agents: int = 2

    # Graph update parameters
    dynamic_obs: bool = True
    obs_radius: int = 10
    full_communication: bool = True
    noise = GaussianNoise()

    # Agent update parameters
    # Possible moves for each drone. Key is the action, value is the (dx, dy) tuple
    possible_moves: {int: int} = {
        0: (0, 0),
        1: (0, 1),
        2: (0, -1),
        3: (-1, 0),
        4: (1, 0),
    }
    empty_cell: int = -1
    global_reward_weight: int = 1
    local_reward_weight: int = 1
    #screen_height: int = 400 
    #screen_width: int = 400 

    def __init__(self):
        self._process_screen_size()
        
    def _process_screen_size(self): 
        height = getattr(self,"screen_height",None) 
        width = getattr(self,"screen_width",None)
            
        if width and height: 
            return 
        
        if not height: 
            height = HEIGHT * (max(self.size // 30, 0) + 1)
            self.screen_height = height

        if not width: 
            width = WIDTH * (max(self.size // 30, 0) + 1)
            self.screen_width = width
        
        return 
    
# Baseline specific configuration parameters
class BaselineConfig(Config):
    D: int = 1

    def __init__(self):
        super().__init__()

# Baseline specific configuration parameters
class EpsilonBaselineConfig(Config):
    D: int = 1
    epsilon: int = 0.2

    def __init__(self):
        super().__init__()
