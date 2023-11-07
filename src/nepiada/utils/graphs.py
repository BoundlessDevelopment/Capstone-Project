# This file implements the communication and observation graphs inspired from Gadjov and Pavel et. al.

import numpy as np
from utils.config import Config
from utils.agent import Agent, AgentType

class Graph():
    def __init__(self, config):
        print("Graphs have been initialized")

        self.dim = config.size
        self.config = config

        ## An adjacency list for which agent can communicate with each other
        self.comm = {agent: [] for agent in range(config.num_good_agents + config.num_adversarial_agents)}

        ## An adjacency list for which agent can observe each other
        self.obs = {agent: [] for agent in range(config.num_good_agents + config.num_adversarial_agents)}


    def update_graph(self, agents):
        for _, agent in agents.items():
            x_coord = agent.p_pos[0]
            y_coord = agent.p_pos[1]

            # Update observation graph here
            for __, other_agent in agents.items():
                if other_agent.uid != agent.uid:
                    pass
                    # Calculate the distance between them


                    # If within observation radius add them to the graph


            # Update communication graph here
            # Do nothing

        return