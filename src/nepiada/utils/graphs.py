# This file implements the communication and observation graphs inspired from Gadjov and Pavel et. al.
import numpy as np
import matplotlib.pyplot as plt
import pygame
from .anim_consts import *


class Graph:
    def __init__(self, config, agents, screen=None, cell_size=0):
        print("Graphs have been initialized")

        self.dim = config.size
        self.agents = agents
        self.dynamic_obs = config.dynamic_obs
        self.full_communication = config.full_communication
        self.observation_radius = config.obs_radius
        self.num_agents = config.num_good_agents + config.num_adversarial_agents
        self.obs_arrows = []
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.cell_size = cell_size

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

        if self.dynamic_obs:
            for agent_name, agent in agents.items():
                x_coord = agent.p_pos[0]
                y_coord = agent.p_pos[1]

                # Update observation graph here
                for other_agent_name, other_agent in agents.items():
                    if other_agent_name != agent_name:
                        # Calculate the distance between them
                        delta_x = other_agent.p_pos[0] - x_coord
                        delta_y = other_agent.p_pos[1] - y_coord

                        distance = np.sqrt(delta_x**2 + delta_y**2)

                        # If within observation radius add them to the graph
                        if distance < self.observation_radius:
                            self.obs[agent_name].append(other_agent_name)
        else:
            # Do not update the observation graph, because it has been configured to be static
            pass

        # Update communication graph here
        # Nothing to do, as we have defined communication graph to be static

        return

    def _draw_arrow(self, color, start, end, head_size=10):
        pygame.draw.line(self.screen, color, start, end, 1)
        rotation = np.degrees(np.arctan2(start[1] - end[1], end[0] - start[0])) + 90
        pygame.draw.polygon(
            self.screen,
            color,
            (
                (
                    end[0] + head_size * np.sin(np.radians(rotation)),
                    end[1] + head_size * np.cos(np.radians(rotation)),
                ),
                (
                    end[0] + head_size * np.sin(np.radians(rotation - 120)),
                    end[1] + head_size * np.cos(np.radians(rotation - 120)),
                ),
                (
                    end[0] + head_size * np.sin(np.radians(rotation + 120)),
                    end[1] + head_size * np.cos(np.radians(rotation + 120)),
                ),
            ),
        )

    def _draw_agents(self, radius=2):
        # Draw the agents and the observations
        for agent_name, agent in self.agents.items():
            # Convert grid positions to pixel positions for drawing
            agent_pixel_pos = (
                agent.p_pos[0] * self.cell_size + self.cell_size // 2,
                agent.p_pos[1] * self.cell_size + self.cell_size // 2,
            )
            color = BLUE if "truthful" in agent_name else RED
            pygame.draw.circle(
                self.screen, color, agent_pixel_pos, self.cell_size // radius
            )

    def _draw_target(self, radius=4):
        # Convert grid positions to pixel positions for drawing
        target_pos = (
            (self.dim / 2) * self.cell_size + self.cell_size // 2,
            (self.dim / 2) * self.cell_size + self.cell_size // 2,
        )
        color = BLACK
        pygame.draw.circle(
            self.screen, color, target_pos, self.cell_size // radius
        )

    # Function to draw the grid
    def _draw_grid(self):
        self.screen.fill(WHITE)
        for x in range(0, WIDTH, self.cell_size):
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, self.cell_size):
            pygame.draw.line(self.screen, BLACK, (0, y), (WIDTH, y))

    def render_graph(self, type="obs"):
        if self.screen is None:
            print("Not drawing anything ...")
            return
        self._draw_grid()
        self._draw_agents(radius=2)
        self._draw_target(radius=4)

        # Draw the agents and the observations
        observations = self.obs if type == "obs" else self.comm

        for observer_name, observed_list in observations.items():
            for observed_name in observed_list:
                if observed_name in self.agents:
                    observed = self.agents[observed_name]
                    observer = self.agents[observer_name]
                    observed_pixel_pos = (
                        observed.p_pos[0] * self.cell_size + self.cell_size // 2,
                        observed.p_pos[1] * self.cell_size + self.cell_size // 2,
                    )
                    observer_pixel_pos = (
                        observer.p_pos[0] * self.cell_size + self.cell_size // 2,
                        observer.p_pos[1] * self.cell_size + self.cell_size // 2,
                    )

                    self._draw_arrow(
                        BLACK, observer_pixel_pos, observed_pixel_pos, head_size=5
                    )

        # Update the display
        pygame.display.flip()
        pygame.time.delay(500)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        # Cap the frame rate
        self.clock.tick(FPS)

    def reset_graphs(self):
        if self.full_communication:
            all_agents = [agent for agent in self.agents]
            self.comm = {agent: all_agents for agent in self.agents}
        else:
            self.comm = {agent: [] for agent in self.agents}

        self.obs = {agent: [] for agent in self.agents}
