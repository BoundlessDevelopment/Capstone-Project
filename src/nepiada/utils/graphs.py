# This file implements the communication and observation graphs inspired from Gadjov and Pavel et. al.

import matplotlib.pyplot as plt
import numpy as np
import time

# Local imports
from utils.config import Config
from utils.agent import Agent, AgentType


class Graph():
    def __init__(self, config, agents):
        print("Graphs have been initialized")

        self.dim = config.size
        self.agents = agents
        self.dynamic_obs = config.dynamic_obs
        self.full_communication = config.full_communication
        self.observation_radius = config.obs_radius
        self.num_agents = config.num_good_agents + config.num_adversarial_agents

        ## An adjacency list for which agent can communicate with each other
        if self.full_communication:
            all_agents = [agent for agent in self.agents]
            self.comm = {agent: all_agents for agent in self.agents}
        else:
            self.comm = {agent: [] for agent in self.agents}

        ## An adjacency list for which agent can observe each other
        self.obs = {agent: [] for agent in self.agents}


    def update_graphs(self, agents):
        # Update the agents
        self.agents = agents

        # Reset the graph
        self.obs = {agent: [] for agent in agents}

        if (self.dynamic_obs):
            for agent_name, agent in agents.items():
                x_coord = agent.p_pos[0]
                y_coord = agent.p_pos[1]

                # Update observation graph here
                for other_agent_name, other_agent in agents.items():
                    if other_agent_name != agent_name:
                        # Calculate the distance between them
                        delta_x = (other_agent.p_pos[0] - x_coord)
                        delta_y = (other_agent.p_pos[1] - y_coord)

                        distance = np.sqrt(delta_x**2 + delta_y**2)

                        # If within observation radius add them to the graph
                        if (distance < self.observation_radius):
                            self.obs[agent_name].append(other_agent_name)
        else:
            # Do not update the observation graph, because it has been configured to be static
            pass

        # Update communication graph here
        # Nothing to do, as we have defined communication graph to be static

        return


    def render_graph(self, comm=True, obs=True):
        #TODO (Arash): Should be replaced with a better rendering utility, example PyGame
        print("-----------")
        if (obs):
            for agent_name, other_agents in self.obs.items():
                print("Agent", agent_name, " observes: ", other_agents)
        if(comm):
            for agent_name, other_agents in self.comm.items():
                print("Agent", agent_name, " communicates with: ", other_agents)
        print("-----------")


    def reset_graphs(self):
        if self.full_communication:
            all_agents = [agent for agent in self.agents]
            self.comm = {agent: all_agents for agent in self.agents}
        else:
            self.comm = {agent: [] for agent in self.agents}

        self.obs = {agent: [] for agent in self.agents}