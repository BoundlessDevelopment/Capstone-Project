import env.nepiada as nepiada

from utils.config import Config

config = Config()

env = nepiada.parallel_env(config = config)
observations, infos = env.reset()

while env.agents:
    # TODO: POLICY
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()