import functools
import numpy as np

import gymnasium
from gymnasium.spaces import Discrete, Box, Dict, Sequence, Text, Tuple

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, aec_to_parallel, wrappers

from utils.config import Config
from utils.world import World
from utils.agent import AgentType
import pygame

import copy
import string
from collections import defaultdict
import os
import matplotlib.pyplot as plt


def parallel_env(config: Config):
    """
    The env function often wraps the environment in wrappers by default.
    Converts to AEC API then back to Parallel API since the wrappers are
    only supported in AEC environments.
    """
    internal_render_mode = "human"
    env = raw_env(render_mode=internal_render_mode, config=config)
    env = parallel_to_aec(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)
    env = aec_to_parallel(env)
    return env


def raw_env(render_mode=None, config: Config = Config()):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = nepiada(render_mode=render_mode, config=config)
    return env


class nepiada(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "nepiada_v1"}

    def __init__(self, render_mode=None, config: Config = Config()):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        These attributes should not be changed after initialization.
        """

        self.total_agents = config.num_good_agents + config.num_adversarial_agents
        self.config = config
        self.possible_agents = []
        self.render_mode = render_mode

        # Initializing agents and grid
        self.world = World(config)

        # Create a folder called plots to save the simulation plots
        os.makedirs(config.simulation_dir, mode=0o777, exist_ok=True)

        # Add agent_names to possible agents
        # Note that each name is unique and hence is an ID
        for agent_name in self.world.agents:
            self.possible_agents.append(agent_name)

    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        ## THANOS EXPERIMENTAL ##
        return Dict(
            {
                "position": Box(
                    low=0, high=self.config.size, shape=(2,), dtype=np.float32
                ),
                "target_neighbours": Sequence(
                    Tuple(
                        Text(min_length=1, max_length=4, charset=string.digits),
                        Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                    )
                ),
                "true_obs": Sequence(
                    Tuple(
                        Text(min_length=1, max_length=4, charset=string.digits),
                        Box(low=0, high=self.config.size, shape=(2,), dtype=np.float32),
                    )
                ),
            }
        )

    # Action space should be defined here.
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Discrete movement, either up, down, stay, left or right.
        return Discrete(len(self.config.possible_moves))

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        elif self.render_mode == "human":
            self.world.graph.render_graph(type="obs")
            return

    def observe(self, agent_name):
        """
        Observe should return the agents within the observation radius of the specified agent.
        Param: Unique agent string identifier: e.g. 'adversarial_0'
        Return: An array of agents who's position it can directly observe
        """
        return np.array(self.world.graph.obs[agent_name])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pygame.quit()
        # Before quiting write out the graphs for each trajectory
        self._dump_pos_graphs()
        pass

    ## THANOS EXPERIMENTAL
    def get_observations(self):
        """
        Experimental // Experimental
        """
        observations = {agent: None for agent in self.agents}
        for agent_name in self.agents:
            observation = {}

            # Get the agent
            curr_agent = self.world.get_agent(agent_name)
            # Write position of the agent in np array format
            observation["position"] = np.array(curr_agent.p_pos, dtype=np.float32)

            # True positions in np array format
            true_pos = []
            for observed_agent_name in self.agents:
                observed_agent = self.world.get_agent(observed_agent_name)
                if observed_agent_name in self.world.graph.obs[agent_name]:
                    true_pos.append(
                        (
                            observed_agent_name,
                            np.array(observed_agent.p_pos, dtype=np.float32),
                        )
                    )
            observation["true_obs"] = true_pos

            # Can add target neighbours here if desired.

            observations[agent_name] = observation
        return observations

    ## THANOS EXPERIMENTAL
    def get_observations_old(self):
        """
        The 2xNxN observation structure returned below are the coordinates of each agents that each agent can directly observe
        observations[i][j] is the location that drone i sees drone j at
        """
        observations = {agent: None for agent in self.agents}
        for agent_name in self.agents:
            observation = {}
            for observed_agent_name in self.agents:
                observed_agent = self.world.get_agent(observed_agent_name)
                if observed_agent_name in self.world.graph.obs[agent_name]:
                    observation[observed_agent_name] = (
                        observed_agent.p_pos[0],
                        observed_agent.p_pos[1],
                    )
                else:
                    observation[observed_agent_name] = None  # Cannot be observed
            observations[agent_name] = observation
        return observations

    def get_all_messages(self):
        """
        The 2xNxNxN message structure returned below are the coordinates that each drone receives from a drone about another drone
        observations[i][j][k] is the location that drone i is told by drone k where drone j is
        """
        incoming_all_messages = {}
        for agent_name in self.agents:
            observation = self.observations[agent_name]

            incoming_agent_messages = {}

            for target_agent_name in self.agents:
                incoming_communcation_messages = {}

                if not observation[target_agent_name]:
                    # Must estimate where the agent is via communication

                    for helpful_agent in self.world.graph.comm[agent_name]:
                        curr_agent = self.world.get_agent(helpful_agent)
                        if curr_agent.type == AgentType.ADVERSARIAL:
                            helpful_beliefs = self.config.noise.add_noise(
                                curr_agent.beliefs
                            )
                        else:
                            helpful_beliefs = curr_agent.beliefs
                        if helpful_beliefs[target_agent_name]:
                            incoming_communcation_messages[helpful_agent] = (
                                helpful_beliefs[target_agent_name][0],
                                helpful_beliefs[target_agent_name][1],
                            )

                incoming_agent_messages[
                    target_agent_name
                ] = incoming_communcation_messages

            incoming_all_messages[agent_name] = incoming_agent_messages
        return incoming_all_messages

    def initialize_beliefs(self):
        """
        Initializing the 2xN structure holds where each agent believes that itself and each other agent is located
        """
        for agent_name in self.agents:
            beliefs = self.world.get_agent(agent_name).beliefs
            for target_agent_name in self.agents:
                beliefs[target_agent_name] = None

    def initialize_infos_with_agents(self):
        for agent_name in self.agents:
            self.infos[agent_name]["agent_instance"] = self.world.get_agent(agent_name)

    def _update_agents_pos(self):
        for agent in self.agents:
            latest_pos = self.world.get_agent(agent_name=agent).p_pos
            self.agents_pos[agent]["p_pos"].append(latest_pos)
            self.agents_pos[agent]["target_dist"].append(
                self.world.get_target_distance(latest_pos)
            )

    def _dump_pos_graphs(self):
        for agent_name, p_pos_dict in self.agents_pos.items():
            all_pos = p_pos_dict["target_dist"]
            # all_dists = p_pos_dict['target_dist']
            plt.figure(figsize=(10, 6))  # You can adjust the figure size
            plt.plot(all_pos, label=f"Distance of {agent_name}", marker="o")
            plt.xlabel("Steps")
            plt.ylabel("Distance to Target")
            plt.title(f"Distance Trajectory of {agent_name}")
            plt.legend()
            plt.grid(True)  # Adds a grid for better readability
            plt.savefig(f"{self.config.simulation_dir}/{agent_name}_traj.png")
            plt.close()  # Close the plot to free up memory

        plt.figure(figsize=(10, 6))  # You can adjust the figure size

        for agent_name, p_pos_dict in self.agents_pos.items():
            all_pos = p_pos_dict["target_dist"]
            # all_dists = p_pos_dict['target_dist']
            plt.plot(all_pos, label=f"Distance of {agent_name}", marker="x")
            plt.grid(True)  # Adds a grid for better readability

        plt.xlabel("Steps")
        plt.ylabel("Distance to Target")
        plt.title(f"Evolution of Agent Distances to Target")
        plt.legend()
        plt.savefig(f"{self.config.simulation_dir}/all_traj.png")
        plt.close()  # Close the plot to free up memory

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        steps taken in the environment.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        print("NEPIADA INFO: All Agents: ", str(self.agents))

        # A list for each agent to show distance from final target
        self.agents_pos = defaultdict(lambda: defaultdict(list))

        self._update_agents_pos()

        self.num_moves = 0

        # Reinitialize the grid
        self.world.grid.reset_grid()

        # Reset the comm and observation graphs
        self.world.graph.reset_graphs()

        # Reset the rewards
        self.rewards = {agent: 0 for agent in self.agents}

        # Reset the truncations
        self.truncations = {agent: False for agent in self.agents}

        # Info will be used to pass information about comm graphs, beliefs, and incoming messages
        self.infos = {agent: {} for agent in self.agents}

        # Initialize the infos with the agent instances, so the algorithm can access AND update beliefs.
        ## THANOS EXPERIMENTAL
        # self.initialize_infos_with_agents()

        # The observation structure returned below are the coordinates of each agents that each agent can directly observe
        self.observations = self.get_observations()

        self.initialize_beliefs()

        print("NEPIADA INFO: Environment Reset Successful. All Checks Passed.")
        return self.observations, self.infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
            The observation structure returned below are the coordinates of each agents that each agent can directly observe

        - rewards
            A 1xN reward vector where agent's uid corresponds to its reward index

        - terminations
            A 1xN vector where each index corresponds to agent's uid.
            True if a particular agent's episode has terminated, false otherwise

        - truncations
            #TODO

        - infos
            Is a dictionary with agent_names as the key. Each value in turn is a dict
            of the form {"comm": [], "beliefs": [], "incoming_all_messages": []}
            To access a agent's communcation graph: infos[agent_name]["comm"]
            To access a agent's incoming_all_messages dictionary: infos[agent_name]["incoming_all_messages"]

        """
        # Assert that number of actions are equal to the number of agents
        assert len(actions) == len(self.agents)

        # Update drone positions
        self.move_drones(actions)

        # Update the running lists keeping track of positions
        self._update_agents_pos()

        # Get the updated rewards
        # TODO: Account for large negative rewards for collision or off-boundary moves
        self.rewards = self.get_rewards()

        terminations = {agent: False for agent in self.agents}
        self.num_moves += 1
        env_truncation = self.num_moves >= self.config.iterations
        truncations = {agent: env_truncation for agent in self.agents}

        # Update the observation and communication graphs at each iteration
        self.world.update_graphs()

        self.observations = self.get_observations()

        # Second pass communicated beliefs
        incoming_all_messages = self.get_all_messages()

        # Info will be used to pass information about comm graphs, beliefs, and incoming messages
        self.infos = {agent_name: {} for agent_name in self.agents}
        for agent_name in self.agents:
            ## THANOS EXPERIMENTAL - This WILL break the baselines
            # self.infos[agent_name]["comm"] = self.world.graph.comm[agent_name]
            self.infos[agent_name]["incoming_messages"] = incoming_all_messages[
                agent_name
            ]
            # self.infos[agent_name]["beliefs"] = self.world.get_agent(agent_name).beliefs
            # self.infos[agent_name]["agent_instance"] = self.world.get_agent(agent_name)

        if self.render_mode == "human":
            self.render()

        return self.observations, self.rewards, terminations, truncations, self.infos

    def move_drones(self, actions):
        """
        Move the drones according to the actions
        """
        # Drones that collided with another drone, for reprocessing
        collided_drones = []

        for agent_name in self.agents:
            agent = self.world.agents[agent_name]
            action = actions[agent_name]
            status = self.world.grid.move_drone(agent, action)
            if status == -2:
                collided_drones.append(agent_name)
            elif status == -1:
                # Drone collided with boundary
                pass

        if collided_drones:
            assert (
                False
            ), "We should be allowing for multiple drones to occupy the same position"

    def get_rewards(self):
        """
        This function assigns reward to all agents based on the following two criterias:

        - global_arrangement_reward : The average distance of all agents from the target
        - local_arrangement_reward : The deviation of the agent from the ideal arrangement with it's target neighbours

        Refer to D. Gadjov and Pavel's paper for more details about it.

        Returns: A dictionary with agent_name as key and reward as a value
        """
        rewards = {}

        # Get the average vector distance of the agents from target
        target_x = self.config.size / 2
        target_y = self.config.size / 2
        global_arrangement_vector = np.array([0.0, 0.0])

        for agent_name in self.agents:
            agent = self.world.agents[agent_name]
            global_arrangement_vector += np.array(
                [(agent.p_pos[0] - target_x), (target_y - agent.p_pos[1])]
            )

        global_arrangement_vector = np.divide(
            global_arrangement_vector, len(self.agents)
        )
        global_arrangement_reward = np.sqrt(
            global_arrangement_vector[0] ** 2 + global_arrangement_vector[1] ** 2
        )

        # We update the global arrangement vector in the graph to visually inspect the global centroid of the agents
        self.world.graph.global_arrangement_vector = global_arrangement_vector

        # Add each agents reward based on their target neighbours
        for agent_name in self.agents:
            agent = self.world.agents[agent_name]
            agent_x = agent.p_pos[0]
            agent_y = agent.p_pos[1]
            deviation_from_arrangement = 0
            for neighbour_name, ideal_distance in agent.target_neighbour.items():
                neighbour = self.world.agents[neighbour_name]
                neighbour_x = neighbour.p_pos[0]
                neighbour_y = neighbour.p_pos[1]
                ideal_x = ideal_distance[0]
                ideal_y = ideal_distance[1]

                deviation_from_arrangement += np.sqrt(
                    (neighbour_x - agent_x - ideal_x) ** 2
                    + (neighbour_y - agent_y - ideal_y) ** 2
                )

            # Compute the agent's net reward, note the negative sign
            rewards[agent_name] = -(
                (self.config.global_reward_weight * global_arrangement_reward)
                + (self.config.local_reward_weight * deviation_from_arrangement)
            )

        return rewards
