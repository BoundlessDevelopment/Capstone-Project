from utils.grid import Grid
from utils.config import Config
from utils.agent import Agent, AgentType

class World():
    def __init__(self):
        print("World has been initialized")

        # Initialize the agents
        self.num_agents = Config.num_good_agents + Config.num_adversarial_agents
        self.agents = []
        for i in range(self.num_agents):
            if i < Config.num_adversarial_agents:
                self.agents.append(Agent(AgentType.ADVERSARIAL))
            else:
                self.agents.append(Agent(AgentType.TRUTHFUL))

        # Initialize the Grid
        self.grid = Grid()

        # Update the grid with agents position
        self.grid.update_grid(self.agents)
        self.grid.print_grid()

        ## The target where all the drones want to reach
        self.target_x = Config.size / 2
        self.target_y = Config.size / 2

    # return all entities in the world
    @property
    def entities(self):
        return self.agents