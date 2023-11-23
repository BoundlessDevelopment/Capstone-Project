import numpy as np
import matplotlib.pyplot as plt 
from IPython.display import display, clear_output
from .agent import AgentType

class Grid():
    def __init__(self, config):
        print("Grid has been initialized")

        self.dim = config.size
        self.config = config
        _, self.state_ax = plt.subplots(figsize=(5, 5))
        plt.ion()

        self.state = np.full((self.dim, self.dim), config.empty_cell, dtype=int)
        self.uid_to_type = {}

    def save_agent_types(self,agents): 
        for agent_name, agent in agents.items(): 
            self.uid_to_type[agent.uid] = agent.type 

            print(agent.uid)

    def get_cell_size(self,width): 
     return width // self.dim

    def update_grid(self, agents):
        #TODO (Arash): Should be replaced with a better rendering utility, example PyGame
        for _, agent in agents.items():
            x_coord = agent.p_pos[0]
            y_coord = agent.p_pos[1]
            # commenting this out for now until we have a better way to render
            # self.state[x_coord][y_coord] = agent.uid if agent.type == AgentType.TRUTHFUL else -agent.uid
            self.state[x_coord][y_coord] = agent.uid

    def reset_grid(self):
        self.state = np.full((self.dim, self.dim), self.config.empty_cell, dtype=int)
    
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
        if self.state[new_x_coord][new_y_coord] != self.config.empty_cell:
            return -2

        # Update the grid
        self.state[x_coord][y_coord] = self.config.empty_cell

        self.state[new_x_coord][new_y_coord] = agent.uid

        # Update the agent position
        agent.p_pos = (new_x_coord, new_y_coord)

        return 0