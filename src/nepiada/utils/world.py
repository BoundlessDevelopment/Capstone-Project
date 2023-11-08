from utils.grid import Grid
from utils.graphs import Graph
from utils.agent import Agent, AgentType


class World:
    def __init__(self, config):
        print("World has been initialized")

        # Initialize the agents
        self.num_agents = config.num_good_agents + config.num_adversarial_agents
        self.agents = {}
        for i in range(self.num_agents):
            if i < config.num_adversarial_agents:
                self.agents["adversarial_" + str(i)] = Agent(AgentType.ADVERSARIAL)
            else:
                self.agents["truthful_" + str(i)] = Agent(AgentType.TRUTHFUL)
        
        # Initialize the Grid
        self.grid = Grid(config)

        # Initialize the graphs
        self.graph = Graph(config, self.agents)

        # Update the grid with agent's position
        self.grid.update_grid(self.agents)
        self.grid.render_grid()

        # Update the graphs with agent's position
        self.graph.update_graphs(self.agents)

        ## The target where all the drones want to reach
        self.target_x = config.size / 2
        self.target_y = config.size / 2

    def update_graphs(self):
        # Update the graphs based on agent's true current position
        self.graph.update_graphs(self.agents)

    def get_agent(self, agent_name):
        return self.agents[agent_name]

    # return all entities in the world
    @property
    def entities(self):
        return self.agents
