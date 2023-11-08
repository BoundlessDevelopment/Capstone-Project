# Local imports
import env.nepiada as nepiada
from utils.config import Config

config = Config()

env = nepiada.parallel_env(config = config)
observations, infos = env.reset()

while env.agents:
    # TODO: POLICY
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    # Note that the observations here are the true position of the agents in the environment
    # They can be accessed by observations[agent_name] = [x_coord, y_coord]
    # The communication and observation graph are stored within info
    # They can be accessed by info[agent_name] = {"obs": [], "comm": []}
    observations, rewards, terminations, truncations, info = env.step(actions)

env.close()