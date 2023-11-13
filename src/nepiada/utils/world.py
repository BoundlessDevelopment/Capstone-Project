# Local imports
from utils.grid import Grid
from utils.graphs import Graph
from utils.agent import Agent, AgentType
import pygame
import numpy as np
from .anim_consts import * 

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
        
        self.screen = self._init_pygame()
        # Initialize the Grid
        self.grid = Grid(config)
        self.grid.save_agent_types(self.agents)
        cell_size = self.grid.get_cell_size(WIDTH)


        # Initialize the graphs
        self.graph = Graph(config, self.agents,screen=self.screen, cell_size=cell_size)
        self.graph.screen = self.screen 

        # Update the grid with agent's position
        self.grid.update_grid(self.agents)

        # Update the graphs with agent's position
        self.graph.update_graphs(self.agents)

        ## The target where all the drones want to reach
        self.target_x = config.size / 2
        self.target_y = config.size / 2

    def _init_pygame(self): 

        # Initialize Pygame
        pygame.init()

        # Setup the display
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Agent Observations")
        return screen
    def __del__(self): 
        pygame.quit()

    def update_graphs(self):
        # Update the graphs based on agent's true current position
        self.graph.update_graphs(self.agents)

    def get_agent(self, agent_name):
        return self.agents[agent_name]

    # return all entities in the world
    @property
    def entities(self):
        return self.agents
