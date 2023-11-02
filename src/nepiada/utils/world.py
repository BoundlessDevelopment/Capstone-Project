from utils.grid import Grid
from utils.agent import Agent, AgentType


class World:
    def __init__(self, config):
        print("World has been initialized")

        # Initialize the agents
        self.num_agents = config.num_good_agents + config.num_adversarial_agents
        self.agents = {}
        for i in range(self.num_agents):
            if i < config.num_adversarial_agents:
                self.agents["adversarial_" + str(i)] = Agent(
                    AgentType.ADVERSARIAL, i + 1
                )
            else:
                self.agents["truthful_" + str(i)] = Agent(AgentType.TRUTHFUL, i + 1)

        # Initialize the Grid
        self.grid = Grid(config)

        # Update the grid with agents position
        self.grid.update_grid(self.agents)
        self.grid.print_grid()

        ## The target where all the drones want to reach
        self.target_x = config.size / 2
        self.target_y = config.size / 2

    # return all entities in the world
    @property
    def entities(self):
        return self.agents
