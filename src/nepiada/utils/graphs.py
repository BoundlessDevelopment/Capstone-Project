# This file implements the communication and observation graphs inspired from Gadjov and Pavel et. al.

import numpy as np
from utils.config import Config
from utils.agent import Agent, AgentType

class Graph():
    def __init__(self, config):
        print("Graphs have been initialized")

        self.dim = config.size
        self.observation_radius = config.obs_radius
        self.num_agents = config.num_good_agents + config.num_adversarial_agents

        ## An adjacency list for which agent can communicate with each other
        self.comm = {agent: [] for agent in range(self.num_agents)}

        ## An adjacency list for which agent can observe each other
        self.obs = {agent: [] for agent in range(self.num_agents)}


    def update_graphs(self, agents):
        # Reset the graph
        self.obs = {agent: [] for agent in range(self.num_agents)}

        for _, agent in agents.items():
            x_coord = agent.p_pos[0]
            y_coord = agent.p_pos[1]

            # Update observation graph here
            for __, other_agent in agents.items():
                if other_agent.uid != agent.uid:
                    # Calculate the distance between them
                    delta_x = (other_agent.p_pos[0] - x_coord)
                    delta_y = (other_agent.p_pos[1] - y_coord)

                    distance = np.sqrt(delta_x**2 + delta_y**2)

                    # If within observation radius add them to the graph
                    if (distance < self.observation_radius):
                        self.obs[agent.uid].append(other_agent.uid)


            # Update communication graph here
            # Nothing to do

        return

    def render_graph(self):
        #TODO: Implement visuals
        pass

    def reset_graphs(self):
        self.comm = {agent: [] for agent in range(config.num_good_agents + config.num_adversarial_agents)}
        self.obs = {agent: [] for agent in range(config.num_good_agents + config.num_adversarial_agents)}
