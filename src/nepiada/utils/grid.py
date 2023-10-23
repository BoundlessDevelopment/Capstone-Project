import numpy as np
from utils.config import Config
from utils.agent import Agent, AgentType

class Grid():
    def __init__(self):
        print("Grid has been initialized")

        self.dim = Config.size

        self.state = np.zeros((self.dim, self.dim))

        ## An adjacency list for which agent can communicate with each other
        ## Will need to build it from config data
        self.Gc = {}

        ## An adjacency list for which agent can observe each other
        ## Will need to build it from config data
        self.Go = {}

    def update_grid(self, agents):
        for agent in agents:
            x_coord = agent.p_pos[0]
            y_coord = agent.p_pos[1]
            self.state[x_coord][y_coord] = agent.type.value

    ## For debug
    def print_grid(self):
        print("---------------")
        for row in self.state:
            print(row)
        print("---------------")