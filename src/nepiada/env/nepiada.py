import functools

import gymnasium
from gymnasium.spaces import Discrete, Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, aec_to_parallel, wrappers

def parallel_env(amount_of_agents, grid_size):
    """
    The env function often wraps the environment in wrappers by default.
    Converts to AEC API then back to Parallel API since the wrappers are
    only supported in AEC environments.
    """
    internal_render_mode = "human"
    env = raw_env(render_mode=internal_render_mode, amount_of_agents=amount_of_agents, grid_size=grid_size)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)
    env = aec_to_parallel(env)
    return env


def raw_env(render_mode=None, amount_of_agents=5, grid_size=10):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = nepiada(render_mode=render_mode, amount_of_agents=amount_of_agents, grid_size=grid_size)
    env = parallel_to_aec(env)
    return env

class nepiada(ParallelEnv):
    def __init__(self, render_mode=None, amount_of_agents=5, grid_size=10):
        """
        Initalizes the environment.
        """
        # TODO: Need to initialize agents
        self.possible_agents = [i for i in range(amount_of_agents)]
        self.render_mode = render_mode

    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # TODO(thanos): define observation space
        return Box()

    # Action space should be defined here.
    # If spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Discrete movement, either up, down, stay, left or right.
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
        # TODO: Reset observations
        observations = observations
        infos = {agent: {} for agent in self.agents}
        self.state = observations

        return observations, infos

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

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {}

        terminations = {agent: False for agent in self.agents}
        self.num_moves += 1
        truncations = {agent: False for agent in self.agents}

        # TODO: Update observation
        observations = observations
        self.state = observations

        # Thanos: Uses infos for communications?
        infos = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos