# Local imports
import env.nepiada as nepiada
from utils.config import BaselineConfig
import pygame


def create_beliefs_with_obs(agent_name, agent_instance, observations, all_agents):
    """
    Updating the agent's current beliefs using observation information
    """
    beliefs = {}

    for agent_name in all_agents:
        if observations[agent_name]:
            beliefs[agent_name] = observations[agent_name]
        else:
            beliefs[agent_name] = None

    beliefs[agent_name] = agent_instance.p_pos

    return beliefs


def strip_extreme_values_and_update_beliefs(
    D_value, incoming_messages, curr_beliefs, new_beliefs, agent_name, all_agents
):
    """
    This function strips the extreme values from the incoming messages according
    to the D value. It strips the D greater values compared to it's current beliefs,
    as well as the D lesser values compared to it's current beliefs and updates the beliefs
    with the average of the remaining communication messages. If no communication messages
    are left for the remaining agents, the agent's new belief of the target's agents position
    remains unchanged.
    """
    for current_agent in all_agents:
        if current_agent == agent_name or new_beliefs[current_agent] is not None:
            continue

        in_messages = []

        # Get incoming messages that contain this agent's position
        for _, comm_messages in incoming_messages.items():
            if comm_messages[current_agent] is not None:
                in_messages.append(comm_messages[current_agent])

        # No communications about this agent, retain previous belief
        if len(in_messages) == 0:
            new_beliefs[current_agent] = curr_beliefs[current_agent]
            continue

        # Strip the extreme values


def step(agent_name, agent_instance, observations, infos, env, config):
    new_beliefs = create_beliefs_with_obs(
        agent_name,
        agent_instance,
        observations,
        env.agents,
        env.action_space(agent_name),
    )
    strip_extreme_values_and_update_beliefs(
        config.D,
        infos["incoming_messages"],
        agent_instance.beliefs,
        new_beliefs,
        agent_name,
        env.agents,
    )


def main():
    env_config = BaselineConfig()

    env = nepiada.parallel_env(config=env_config)
    observations, infos = env.reset()

    while env.agents:
        actions = {}
        for agent in env.agents:
            curr_agent = infos[agent]["agent_instance"]
            agent_action = step(
                agent, curr_agent, observations[agent], infos[agent], env, env_config
            )
            actions[agent] = agent_action

        # Note that the observations here are the true position of the agents in the environment
        # They can be accessed by observations[agent_name] = [x_coord, y_coord]
        # The communication and observation graph are stored within info
        # They can be accessed by info[agent_name] = {"obs": [], "comm": []}
        observations, rewards, terminations, truncations, info = env.step(actions)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
