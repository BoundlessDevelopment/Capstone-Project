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
from collections import OrderedDict, defaultdict
import os
import matplotlib.pyplot as plt

from utils.online_k import *

TYPE_CHECK = True

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
        
        print("NEPIADA INFO: All Agents: ", str(self.possible_agents))

    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        This is the way the observations are structured for RLib. Eventually all the values are flattened internally.
        Note that the order in which the observation is flattened is alphabetically in order of the key values.
        """
        return Dict(
            {
                "target_neighbours": Box(
                    low=-self.config.size, high=self.config.size, shape=(self.total_agents, 2), dtype=np.float32
                ),
                "beliefs": Box(
                    low=0,
                    high=self.config.size + 1,
                    shape=(self.total_agents, 2),
                    dtype=np.float32,
                ),
                "agent_position": Box(
                    low=0, high=self.config.size + 1, shape=(2,), dtype=np.float32
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


    def strip_extreme_values_and_update_beliefs(
        self, incoming_messages, curr_beliefs, new_beliefs, target_agent_name
    ):
        """
        This function strips the extreme values from the incoming messages according
        to the D value. It strips the D greater values compared to it's current beliefs,
        as well as the D lesser values compared to it's current beliefs and updates the beliefs
        with the average of the remaining communication messages. If no communication messages
        are left for the remaining agents, the agent's new belief of the target's agents position
        remains unchanged.
        """
        D_value = self.config.D

        # Estimate the postion of the target agent based on the incomming messages from other agents
        in_messages = []
        for other_agent in incoming_messages:
            if target_agent_name in incoming_messages[other_agent]:
                message = incoming_messages[other_agent][target_agent_name]
                
                # Type check
                if TYPE_CHECK and not isinstance(message, (np.ndarray, np.generic)):
                    print("Found type: ", type(message))
                    assert False, "The type of helpful belief is not a numpy array"

                in_messages.append(message)

        # If we received no information about the target agent we use the previous information
        if len(in_messages) == 0:
            new_beliefs[target_agent_name] = curr_beliefs[target_agent_name]

            # Type check
            if TYPE_CHECK and not isinstance(curr_beliefs[target_agent_name], (np.ndarray, np.generic)):
                print("Found type: ", type(curr_beliefs[target_agent_name]))
                assert False, "The type of net_estimate when no info is available is not a numpy array"
            
        else:
            if curr_beliefs[target_agent_name] is None:
                # Average all the incoming messages for the case where we don't have an estimate for the current agent
                x_pos_mean = sum([message[0] for message in in_messages]) / len(in_messages)
                y_pos_mean = sum([message[1] for message in in_messages]) / len(in_messages)
                new_beliefs[target_agent_name] = np.array([x_pos_mean, y_pos_mean], dtype=np.float32)
                return

            if len(in_messages) <= D_value * 2:
                # Not enough messages to strip
                if TYPE_CHECK and not isinstance(curr_beliefs[target_agent_name], (np.ndarray, np.generic)):
                    print("Found type: ", type(curr_beliefs[target_agent_name]))
                    assert False, "The type of net_estimate when no info is available is not a numpy array"

                new_beliefs[target_agent_name] = np.array(curr_beliefs[target_agent_name], dtype=np.float32)
                return

            x_pos_deviation = []
            y_pos_deviation = []
            for message in in_messages:
                x_pos_deviation.append(message[0] - curr_beliefs[target_agent_name][0])
                y_pos_deviation.append(message[1] - curr_beliefs[target_agent_name][1])

            # Sort the deviations
            x_pos_deviation.sort()
            y_pos_deviation.sort()

            # Remove D lowest and D highest values
            x_pos_deviation = x_pos_deviation[D_value:-D_value]
            y_pos_deviation = y_pos_deviation[D_value:-D_value]

            # Average the remaining values
            x_pos_delta = sum(x_pos_deviation) / len(x_pos_deviation)
            y_pos_delta = sum(y_pos_deviation) / len(y_pos_deviation)

            # Update the beliefs
            new_beliefs[target_agent_name] = np.array([
                curr_beliefs[target_agent_name][0] + x_pos_delta,
                curr_beliefs[target_agent_name][1] + y_pos_delta,
            ], dtype=np.float32)

    def get_observations(self, incoming_messages):
        """
        The observation space is an OrderedDict where the key is the agent_name and the value is a dictionary of two types of observations:

        - agent_position: The current position of the agent
        - beliefs: The belief each agent has about every other agents position
            - This is made by first getting info about observable agents
            - Then for the rest of the agents we estimate using the communicated messages
        """
        beliefs = {agent: None for agent in self.agents}

        # Get the actual position of agents within the observation radius
        for agent_name in self.agents:
            agent_beliefs = {}
            for observed_agent_name in self.agents:
                observed_agent = self.world.get_agent(observed_agent_name)
                if (observed_agent_name == agent_name) or (observed_agent_name in self.world.graph.obs[agent_name]):
                    agent_beliefs[observed_agent_name] = observed_agent.p_pos # Can be observed
                else:
                    agent_beliefs[observed_agent_name] = None  # Cannot be observed
            beliefs[agent_name] = agent_beliefs

        # Estimate the position of the remaining agents using comm graph and extrema pruning
        for agent_name in self.agents:
            agent = self.world.get_agent(agent_name)

            for other_agent_name in self.agents:
                # The other agent is either itself or observed
                if (agent == other_agent_name or beliefs[agent_name][other_agent_name] is not None):
                    continue

                if incoming_messages[agent_name] is not None:
                    self.strip_extreme_values_and_update_beliefs(
                        incoming_messages[agent_name],
                        agent.beliefs,
                        beliefs[agent_name],
                        other_agent_name
                    )
                else:
                    assert False, "Logically, we should never reach here"
                    print("Agent not within communication or observation radius!")

        # Sanity check
        for agent_name in self.agents:
            if agent_name not in beliefs or beliefs[agent_name] is None:
                assert False, "By this point none of the beliefs should be None"
            for other_agent_name in self.agents:
                if other_agent_name not in beliefs[agent_name] or beliefs[agent_name][other_agent_name] is None:
                    assert False, "By this point none of the beliefs should be None"
                if TYPE_CHECK and not isinstance(beliefs[agent_name][other_agent_name], (np.ndarray, np.generic)):
                    print("Found type: ", type(beliefs[agent_name][other_agent_name]))
                    print("The entry: ", (beliefs[agent_name][other_agent_name]))
                    assert False, "The final state of belief is not a numpy array"

        # Update the beliefs of the agents
        for agent_name in self.agents:
            agent = self.world.get_agent(agent_name)
            agent.beliefs = beliefs[agent_name]

        # RLib Observations
        observations = {agent: None for agent in self.agents}
        for agent_name in self.agents:
            observation = OrderedDict()
            # Get the agent
            curr_agent = self.world.get_agent(agent_name)
            # Write position of the agent in np array format
            observation["agent_position"] = np.array(curr_agent.p_pos, dtype=np.float32)

            # True positions in np array format
            final_beliefs = []
            for other_agent_name in self.agents:
                final_beliefs.append(beliefs[agent_name][other_agent_name])
            
            observation["beliefs"] = np.array(final_beliefs, dtype=np.float32)
            # Clip the beliefs to be within the grid
            observation["beliefs"] = np.clip(observation["beliefs"], 0, self.config.size + 1)

            # Store the target neighbours
            observation["target_neighbours"] = self.obs_target_neighbours[agent_name]
            observations[agent_name] = observation

        # Sanity Check
        for agent, observation in observations.items():
            if TYPE_CHECK and not isinstance(observation["agent_position"], (np.ndarray, np.generic)):
                print("Found position type: ", type(observation["agent_position"]))
                print("The entry: ", (observation["agent_position"]))
                assert False, "The position in RLib observation is not a numpy array"

            if TYPE_CHECK and not isinstance(observation["beliefs"], (np.ndarray, np.generic)):
                print("Found beliefs type: ", type(observation["beliefs"]))
                print("The entry: ", (observation["beliefs"]))
                assert False, "The beliefs in RLib observation is not a numpy array"
            
        # print(f"RLib observation: {observations}\n")
        return observations

    def get_all_messages(self):
        """
        The 2xNxNxN message structure returned below are the coordinates that each drone receives from a drone about another drone
        observations[i][j][k] is the location that drone i is told by drone k where drone j is
        """
        incoming_all_messages = {}
        for agent_name in self.agents:
            observation = self.world.graph.obs[agent_name]
            incoming_agent_messages = {}

            for target_agent_name in self.agents:
                incoming_communcation_messages = {}

                if target_agent_name not in observation:
                    # Must estimate where the agent is via communication
                    for helpful_agent in self.world.graph.comm[agent_name]:
                        curr_agent = self.world.get_agent(helpful_agent)
                        if curr_agent.type == AgentType.ADVERSARIAL:
                            helpful_beliefs = self.config.noise.add_noise(curr_agent.beliefs)
                        else:
                            helpful_beliefs = curr_agent.beliefs

                        # Type check
                        for key, value in helpful_beliefs.items():
                            if TYPE_CHECK and not isinstance(value, (np.ndarray, np.generic)):
                                print("Found type: ", type(value))
                                assert False, "The type of helpful belief is not a numpy array"

                        if helpful_beliefs[target_agent_name] is not None:
                            incoming_communcation_messages[helpful_agent] = helpful_beliefs[target_agent_name]

                incoming_agent_messages[target_agent_name] = incoming_communcation_messages

            incoming_all_messages[agent_name] = incoming_agent_messages

        for agent_name in self.agents:
            curr_agent = self.world.get_agent(agent_name)
            for talking_agent in self.agents:
                incoming_messages = []

                for target_agent in self.agents:
                    # Check if the keys exist in the nested dictionary
                    if agent_name in incoming_all_messages and \
                    target_agent in incoming_all_messages[agent_name] and \
                    talking_agent in incoming_all_messages[agent_name][target_agent]:

                        message = incoming_all_messages[agent_name][target_agent][talking_agent]
                    else:
                        # Handle the case where the key doesn't exist
                        # This could be a default value or a special indicator
                        message = None  # or some default value

                    incoming_messages.append(message)

                past = 5
                agents = len(self.agents)
                if(talking_agent not in curr_agent.last_messages):
                    curr_agent.last_messages[talking_agent] = [None]*(agents*(past-1))

                curr_agent.last_messages[talking_agent].extend(incoming_messages)
                if(len(curr_agent.last_messages[talking_agent]) > agents*past):
                    for i in range(agents):
                        curr_agent.last_messages[talking_agent].pop(0)

        #generate truthful weights based on k-means classification
        for agent_name in self.agents:
            curr_agent = self.world.get_agent(agent_name)
            curr_agent.truthful_weights = []
            for target_agent in self.agents:
                example_input = curr_agent.last_messages[target_agent]
                valid_intervals = all(any(x is not None for x in example_input[i:i+9]) for i in range(0, len(example_input), 9))
                if valid_intervals:
                    data_point = np.array([calculate(example_input)])
                    curr_agent.model.partial_fit(data_point.reshape(-1, 1))
                    predicted_cluster = curr_agent.model.predict(data_point.reshape(-1, 1))[0]
                    curr_agent.truthful_weights.append(predicted_cluster)
                else:
                    curr_agent.truthful_weights.append(0.5) #update with midpoint 0.5 when unsure 
            for i in range(len(curr_agent.truthful_weights)):
                if curr_agent.truthful_weights[i] == 0:
                    curr_agent.truthful_weights[i] = 0.1
            print(curr_agent.truthful_weights)
        return incoming_all_messages


    def initialize_beliefs(self):
        """
        Initializing the 2xN structure holds where each agent believes that itself and each other agent is located
        """
        for agent_name in self.agents:
            agent = self.world.get_agent(agent_name)
            for target_agent_name in self.agents:
                if (target_agent_name == agent_name):
                    agent.beliefs[target_agent_name] = np.array(agent.p_pos, dtype=np.float32)
                else:
                    agent.beliefs[target_agent_name] = np.array([np.random.randint(self.config.size), np.random.randint(self.config.size)], dtype=np.float32)

    def _reset_agent_pos(self):
        for agent_name in self.agents:
            self.world.get_agent(agent_name).p_pos = np.random.randint(low=0, high=Config.size, size=2)

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

    def _set_obs_target_neighbours(self):
        # Populate a list of target neighbours for each agent, if there isn't a neighbour relationship, set distance as [0, 0],
        # otherwise set the distance as the relative distance between the two agents
        obs_target_neighbours = {}
        for agent_name in self.agents:
            agent = self.world.get_agent(agent_name)
            target_neighs = []
            for other_agent_name in self.agents:
                if other_agent_name not in agent.target_neighbour:
                    target_neighs.append([0, 0])
                else:
                    target_neighs.append(agent.target_neighbour[other_agent_name])
            obs_target_neighbours[agent_name] = np.array(target_neighs, dtype=np.float32)
        return obs_target_neighbours

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        steps taken in the environment.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]

        # A list for each agent to show distance from final target
        self.agents_pos = defaultdict(lambda: defaultdict(list))

        self._update_agents_pos()

        self.num_moves = 0

        self._reset_agent_pos()

        # Reinitialize the grid
        self.world.grid.reset_grid()
        self.world.grid.update_grid(self.world.agents)

        # Reset the comm and observation graphs
        self.world.graph.reset_graphs()
        self.world.graph.update_graphs(self.world.agents)

        # Store prev scores in agent
        scores = self._compute_scores()
        self._store_scores_in_agent(scores)

        # Reset the rewards
        self.rewards = {agent: 0 for agent in self.agents}

        # Reset the truncations
        self.truncations = {agent: False for agent in self.agents}

        # Infos is used to pass aditional information
        self.infos = {agent: {} for agent in self.agents}

        # Store the target neighbours for each agent
        self.obs_target_neighbours = self._set_obs_target_neighbours()

        # Initialize the infos with the agent instances, so the algorithm can access agent beliefs.
        if (self.config.pass_agents_in_infos):
            self.initialize_infos_with_agents()

        # For incomming messages
        self.incoming_msgs = {agent: {} for agent in self.agents}

        # Set all beliefs to random values
        self.initialize_beliefs()

        # The observation structure returned below are the coordinates of each agents that each agent can directly observe
        self.observations = self.get_observations(self.incoming_msgs)

        # Store the minimum score seen so far, may or may not be used in rewards depending on the function in use.
        self.min_score = min(scores.values())

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
        
        # For incomming messages
        self.incoming_msgs = {agent: {} for agent in self.agents}

        # Info will be used to pass information about comm graphs, beliefs, and incoming messages
        # self.infos = {agent_name: {} for agent_name in self.agents}
        
        # For incomming messages
        self.incoming_msgs = {agent: {} for agent in self.agents}

        incoming_all_messages = self.get_all_messages()
        for agent_name in self.agents: 
            self.incoming_msgs[agent_name] = incoming_all_messages[agent_name]

        # if self.render_mode == "human":
        #     self.render()

        self.observations = self.get_observations(self.incoming_msgs)
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
                # THANOS EXPERIMENTAL - Large negative reward will be given on instance.
                #self.world.agents[agent_name].prev_score = -self.world.agents[agent_name].prev_score
                pass

        if collided_drones:
            assert (
                False
            ), "We should be allowing for multiple drones to occupy the same position"

    ## THANOS EXPERIMENTAL
    ## Rewards - Default
    def get_rewards(self):
        rewards = {}
        curr_scores = self._compute_scores()

        for agent_name in self.agents:
            rewards[agent_name] = curr_scores[agent_name]
        
        self._store_scores_in_agent(curr_scores)
        return rewards

    ## THANOS EXPERIMENTAL
    def get_rewards_no_delta(self):
        rewards = {}
        curr_scores = self._compute_scores()
    
        values = curr_scores.values()
        min_r = min(values)
        if min_r < self.min_score:
            self.min_score = min_r

        if min_r == 0:
            print("NEPIADA WARN: All rewards are the same. This should not happen.")
            print(f"NEPIADA INFO: Current Scores: {str(curr_scores)} | Current Rewards: {str(rewards)} | Values: {str(values)}")

        for agent_name in self.agents:
            if self.world.agents[agent_name].prev_score <= 0:
                # Normalize the rewards to be between 0 and 10
                if self.min_score == 0:
                    rewards[agent_name] = 0
                else:
                    rewards[agent_name] = (curr_scores[agent_name] - self.min_score) / (0 - self.min_score) * 10

                # Boundary penalty, -1 for how close an agent is to the boundary, capped at -10
                dist_to_left = self.world.agents[agent_name].p_pos[0]
                dist_to_right = self.config.size - self.world.agents[agent_name].p_pos[0]
                dist_to_top = self.config.size - self.world.agents[agent_name].p_pos[1]
                dist_to_bottom = self.world.agents[agent_name].p_pos[1]

                min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
                if min_dist <= 6:
                    rewards[agent_name] -= (6 - min_dist)
            else:
                # -10 reward for agents that collided with boundary
                rewards[agent_name] = -10
        
        self._store_scores_in_agent(curr_scores)
        return rewards

    ## THANOS EXPERIMENTAL
    ## Compute Scores with more local-focused rewards
    def _compute_scores(self):
        """
        Compute the scores of the agents based on their distance from the target
        """
        scores = {}
    
        target_x = self.config.size / 2
        target_y = self.config.size / 2
        global_arrangement_vector = np.array([0.0, 0.0])

        # for agent_name in self.agents:
        #     agent = self.world.agents[agent_name]
        #     global_arrangement_vector += np.array(
        #         [(agent.p_pos[0] - target_x), (target_y - agent.p_pos[1])]
        #     )

        # global_arrangement_vector = np.divide(
        #     global_arrangement_vector, len(self.agents)
        # )

        # We update the global arrangement vector in the graph to visually inspect the global centroid of the agents
        self.world.graph.global_arrangement_vector = global_arrangement_vector

        # Add each agents reward based on their target neighbours
        for agent_name in self.agents:
            agent = self.world.agents[agent_name]
            agent_x = agent.p_pos[0]
            agent_y = agent.p_pos[1]
            deviation_from_arrangement = 0
            deviation_from_global_arrangement = np.sqrt((agent_x - target_x) ** 2 + (target_y - agent_y) ** 2)

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

            scores[agent_name] = -((self.config.global_reward_weight * deviation_from_global_arrangement) + (self.config.local_reward_weight * deviation_from_arrangement))
            
        return scores

    ## THANOS EXPERIMENTAL
    def _compute_scores_global(self):
        """
        Compute the scores of the agents based on their distance from the target
        """
        scores = {}
    
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

            scores[agent_name] = -((self.config.global_reward_weight * global_arrangement_reward) + (self.config.local_reward_weight * deviation_from_arrangement))
            
        return scores

    def _store_scores_in_agent(self, scores):
        for agent_name in self.agents:
            self.world.agents[agent_name].prev_score = scores[agent_name]

