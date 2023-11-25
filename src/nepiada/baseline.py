# Local imports
import numpy as np
import env.nepiada as nepiada
from nepiada.utils import agent
from utils.config import BaselineConfig
import pygame


def calculate_cost(agent_name, target_neighbours, beliefs, grid_size):
    """
    This function calculates the cost value given the agent's beliefs.
    Before calling this method, the agent's belief of itself should be updated
    with it's intended new position.

    It is calculated using a similar function to the cost function in
    Diane and Prof. Pavel's Paper, however modified to the discrete space.
    """
    arrangement_cost = 0
    target_neighbor_cost = 0

    target_x = grid_size / 2
    target_y = grid_size / 2

    # Calculate the global arrangement cost
    for curr_agent_name, agent_belief in beliefs.items():
        if agent_belief is None:
            # What do we do here?
            continue
        arrangement_cost += np.sqrt(
            (agent_belief[0] - target_x) ** 2 + (agent_belief[1] - target_y) ** 2
        )

    arrangement_cost /= len(beliefs)

    # Calculate the target neighbour cost
    for curr_agent_name, target_relative_position in target_neighbours.items():
        assert beliefs[agent_name] is not None

        curr_agent_position = beliefs[curr_agent_name]
        agent_position = beliefs[agent_name]

        if curr_agent_position is None:
            # What do we do here?
            continue
        target_neighbor_cost += np.sqrt(
            (curr_agent_position[0] - agent_position[0] - target_relative_position[0])
            ** 2
            + (curr_agent_position[1] - agent_position[1] - target_relative_position[1])
            ** 2
        )

    # Return cost, should this be weighted?
    return arrangement_cost + target_neighbor_cost


def create_beliefs_with_obs(agent_name, observations, all_agents):
    """
    Create a new beliefs dict with the agent's observations filled in as the
    groundtruth. If the agent has no observations of another agent,
    the belief of that agent is set to None.
    """
    beliefs = {}

    for agent_name in all_agents:
        if observations[agent_name]:
            beliefs[agent_name] = observations[agent_name]
        else:
            beliefs[agent_name] = None

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
        if incoming_messages[current_agent] is None:
            # No incoming messages about this agent, keep previous state
            new_beliefs[current_agent] = curr_beliefs[current_agent]
            continue

        in_messages = []

        # Get incoming messages that contain this agent's position
        for _, comm_message in incoming_messages[current_agent].items():
            in_messages.append(comm_message)

        # Strip the extreme values
        x_pos_deviation = []
        y_pos_deviation = []
        for message in in_messages:
            x_pos_deviation.append(message[0] - curr_beliefs[current_agent][0])
            y_pos_deviation.append(message[1] - curr_beliefs[current_agent][1])

        if len(x_pos_deviation) <= D_value * 2 or len(y_pos_deviation) <= D_value * 2:
            # Not enough messages to strip
            new_beliefs[current_agent] = curr_beliefs[current_agent]
            continue

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
        new_beliefs[current_agent] = (
            curr_beliefs[current_agent][0] + x_pos_delta,
            curr_beliefs[current_agent][1] + y_pos_delta,
        )


def step(agent_name, agent_instance, observations, infos, env, config):
    # Create new beliefs dict with observation information
    new_beliefs = create_beliefs_with_obs(
        agent_name,
        observations,
        env.agents,
        env.action_space(agent_name),
    )
    # If there are incoming messages, process them and update beliefs
    if "incoming_messages" in infos:
        strip_extreme_values_and_update_beliefs(
            config.D,
            infos["incoming_messages"],
            agent_instance.beliefs,
            new_beliefs,
            agent_name,
            env.agents,
        )

    # Calculate the cost for every possible action
    action_costs = {}
    for action in env.action_space(agent_name):
        # Calculate the cost for the action
        new_beliefs[agent_name] = (
            agent_instance.p_pos[0] + config.possible_moves[action][0],
            agent_instance.p_pos[1] + config.possible_moves[action][1],
        )
        action_costs[action] = calculate_cost(
            agent_name, agent_instance.target_neighbour, new_beliefs, config.size
        )

    # Choose the action with the lowest cost
    min_action = min(action_costs, key=action_costs.get)
    new_beliefs[agent_name] = (
        agent_instance.p_pos[0] + config.possible_moves[min_action][0],
        agent_instance.p_pos[1] + config.possible_moves[min_action][1],
    )
    agent_instance.beliefs = new_beliefs

    return min_action


def main():
    env_config = BaselineConfig()

    env = nepiada.parallel_env(config=env_config)
    observations, infos = env.reset()

    while env.agents:
        actions = {}
        for c_agent in env.agents:
            curr_agent = infos[c_agent]["agent_instance"]
            agent_action = step(
                c_agent, curr_agent, observations[c_agent], infos[c_agent], env, env_config
            )
            actions[c_agent] = agent_action

        observations, rewards, terminations, truncations, info = env.step(actions)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
