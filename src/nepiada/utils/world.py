# Local imports
from utils.grid import Grid
from utils.graphs import Graph
from utils.agent import Agent, AgentType


class World:
    def __init__(self, config):
        # Check if the number of agents match the agents target size
        assert config.num_good_agents + config.num_adversarial_agents == config.agent_grid_width * config.agent_grid_height

        # Initialize the agents
        self.num_agents = config.num_good_agents + config.num_adversarial_agents
        self.agents, self.agent_uid_to_name = self.__initialize_agents(self.num_agents, config)

        # Set each agent's adjacent target neighbours
        self.__initialize_target_neighbours(self.num_agents, config)
        
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

        print("World has been initialized")

    def __initialize_agents(self, num_agents, config):
        """
            This is a private function

            Here we initialize adversarial agents first followed by truthful agents
            # TODO: They can be randomized by a particular seed

            Returns:
                agents: Dictionary of agent_name to agent object
                agent_uid_to_name: Dictionary of agent_uid to agent_name
        """
        agents = {}
        agent_uid_to_name = {}
        for i in range(num_agents):
            # Initialize the agent
            agent_name = ""
            if i < config.num_adversarial_agents:
                agent_name = "adversarial_" + str(i)
                agents[agent_name] = Agent(AgentType.ADVERSARIAL)
            else:
                agent_name = "truthful_" + str(i)
                agents[agent_name] = Agent(AgentType.TRUTHFUL)

            # Map the uid to agent name for later use
            agent_uid_to_name[i] = agent_name

        return agents, agent_uid_to_name

    def __initialize_target_neighbours(self, num_agents, config):
        """
            This is a private function

            The target neighbours for each agent is its left, right, top and bottom agents,
            if they exist. The agent should try to maintain a unit distance from them.

            Returns: Nothing
        """
        assert len(self.agents) != 0
        assert len(self.agent_uid_to_name) != 0

        for i in range(num_agents):
            # Get the agent
            width = config.agent_grid_width
            height = config.agent_grid_height
            agent_name = self.agent_uid_to_name[i]
            agent = self.agents[agent_name]

            # Check the left
            if (i % width != 0):
                # Add the target neighbour
                neighbour_name = self.agent_uid_to_name[i - 1]
                agent.set_target_neighbour(neighbour_name, [-1, 0])

            # Check the right
            if ((i + 1) % width != 0):
                # Add the target neighbour
                neighbour_name = self.agent_uid_to_name[i + 1]
                agent.set_target_neighbour(neighbour_name, [1, 0])

            # Check the top
            if (i // width != 0):
                # Add the target neighbour
                neighbour_name = self.agent_uid_to_name[i - width]
                agent.set_target_neighbour(neighbour_name, [0, 1])

            # Check the bottom
            if (i // width != height - 1):
                # Add the target neighbour
                neighbour_name = self.agent_uid_to_name[i + width]
                agent.set_target_neighbour(neighbour_name, [0, -1])

        return

    def update_graphs(self):
        # Update the graphs based on agent's true current position
        self.graph.update_graphs(self.agents)

    def get_agent(self, agent_name):
        return self.agents[agent_name]

    # return all entities in the world
    @property
    def entities(self):
        return self.agents
