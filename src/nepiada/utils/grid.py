import numpy as np
from utils.config import Config
from utils.agent import Agent, AgentType

class Grid():
    def __init__(self, config):
        print("Grid has been initialized")

        self.dim = config.size
        self.config = config

        self.state = np.zeros((self.dim, self.dim), dtype=int)

        ## An adjacency list for which agent can communicate with each other
        ## Will need to build it from config data
        self.Gc = {}

        ## An adjacency list for which agent can observe each other
        ## Will need to build it from config data
        self.Go = {}

    def update_grid(self, agents):
        for _, agent in agents.items():
            x_coord = agent.p_pos[0]
            y_coord = agent.p_pos[1]
            # commenting this out for now until we have a better way to render
            # self.state[x_coord][y_coord] = agent.uid if agent.type == AgentType.TRUTHFUL else -agent.uid
            self.state[x_coord][y_coord] = agent.uid
    
    def move_drone(self, agent, action):
        """
        Moves drone in corresponding direction, checks for validity of the move
        """
        # Get the current position of the agent
        x_coord = agent.p_pos[0]
        y_coord = agent.p_pos[1]

        # Check if the move is to stay in the same position, if so just return
        dx = self.config.possible_moves[action][0]
        dy = self.config.possible_moves[action][1]

        if dx == 0 and dy == 0:
            return 0

        # Get the new position of the agent
        new_x_coord = x_coord + dx
        new_y_coord = y_coord + dy

        # Check if the new position is valid
        if new_x_coord < 0 or new_x_coord >= self.dim or new_y_coord < 0 or new_y_coord >= self.dim:
            return -1

        # Check if the new position is occupied
        if self.state[new_x_coord][new_y_coord] != 0:
            return -2

        # Update the grid
        self.state[x_coord][y_coord] = 0
        # commenting this out for now until we have a better way to render
        # self.state[new_x_coord][new_y_coord] = agent.uid if agent.type == AgentType.TRUTHFUL else -agent.uid
        self.state[new_x_coord][new_y_coord] = agent.uid

        # Update the agent position
        agent.p_pos = (new_x_coord, new_y_coord)

        return 0

    ## For debug
    def print_grid(self):
        print("---------------")
        for row in self.state:
            print(row)
        print("---------------")