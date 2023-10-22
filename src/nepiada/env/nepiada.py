import functools
import numpy as np

import gymnasium
from gymnasium.spaces import Discrete, Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, aec_to_parallel, wrappers

from utils.config import Config

def parallel_env(config : Config):
    """
    The env function often wraps the environment in wrappers by default.
    Converts to AEC API then back to Parallel API since the wrappers are
    only supported in AEC environments.
    """
    internal_render_mode = "human"
    env = raw_env(render_mode=internal_render_mode, config = config)
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
    env = nepiada(render_mode=render_mode, config = config)
    env = parallel_to_aec(env)
    return env

class nepiada(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "nepiada_v1"}

    def __init__(self, render_mode=None, config: Config = Config()):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """

        self.total_agents = config.num_good_agents + config.num_adversarial_agents
        self.possible_agents = ["agent_" + str(i) for i in range(self.total_agents)]
        self.render_mode = render_mode
        # self.observations = np.zeros(self.total_agents)

        # TODO: Need to make a grid and initialize agents - currently just adding all agents
        # grid = make_grid(config.size)

    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/

        # Observation space is defined as a N x 2 matrix, where each row corresponds to an agents coordinates.
        # The first column stores the x coordinate and the second column stores the y coordinate
        return Box(low=0, high=config.size, shape=(self.total_agents, 2), dtype=np.int_)

    # Action space should be defined here.
    # If spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Discrete movement, either up, down, stay, left or right.
        # TODO: Make an ENUM for actions
        return Discrete(5)

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

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        steps taken in the environment.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0

        # TODO: Reinitialize the grid

        # Reset the observations
        self.observations = {agent: None for agent in self.agents}

        # Reset the rewards
        self.rewards = {agent: 0 for agent in self.agents}

        # Reset the truncations
        self.truncations = {agent: False for agent in self.agents}

        self.infos = {agent: {} for agent in self.agents}
        print(set(self.infos.keys()))
        print(set(self.agents))

        self.state = self.observations

        return self.observations, self.infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # We should update the rewards here, for now, we will just set everything to 0
        rewards = {agent: 0 for agent in self.agents}

        terminations = {agent: False for agent in self.agents}
        self.num_moves += 1
        env_truncation = self.num_moves >= Config.iterations
        truncations = {agent: env_truncation for agent in self.agents}

        # TODO: Update observation
        observations = self.observations
        self.state = observations

        # Thanos: Uses infos for communications? - How? @HP
        infos = {agent: {} for agent in self.agents}

        # TODO: Need to add information to infos


        if self.render_mode == "human":
            self.render()
        
        return observations, rewards, terminations, truncations, infos