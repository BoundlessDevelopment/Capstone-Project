# Local imports
import env.nepiada as nepiada
from utils.config import Config

config = Config()

env = nepiada.parallel_env(config = config)
observations, infos = env.reset()

def update_beliefs(env):
    """
    CURRENTLY NOT IN USE
    Updating the 2xN structure holds where each agent believes that itself and each other agent is located
    """
    for agent_name in env.agents:
        beliefs = env.world.get_agent(agent_name).beliefs
        observation = env.observations[agent_name]
        for target_agent_name in env.agents:
            if observation[target_agent_name]: # Can directly see
                beliefs[target_agent_name] = observation[target_agent_name]
            else: # Must estimate where the agent is via communication
                beliefs[target_agent_name] = None

while env.agents:
    # TODO: POLICY
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    # Note that the observations here are the true position of the agents in the environment
    # They can be accessed by observations[agent_name] = [x_coord, y_coord]
    # The communication and observation graph are stored within info
    # They can be accessed by info[agent_name] = {"obs": [], "comm": []}
    observations, rewards, terminations, truncations, info = env.step(actions)

env.close()