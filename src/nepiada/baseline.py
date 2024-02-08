# Local imports
import numpy as np
import env.nepiada as nepiada
from utils.config import BaselineConfig
import pygame


def calculate_cost(agent_name, target_neighbours, beliefs, grid_size, config):
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

    length = len(beliefs)
    arrangement_cost_vector = np.array([0.0, 0.0])

    # Calculate the global arrangement cost using vector distances
    for agent_pos in beliefs:
        if agent_pos is None:
            assert False, "Belief can never be None"
            length -= 1
        else:
            arrangement_cost_vector += np.array([(agent_pos[0] - target_x), (target_y - agent_pos[1])])

    # Normalize the global arrangement_cost_vector
    if (length != 0):
        arrangement_cost_vector = np.divide(arrangement_cost_vector, length)

    arrangement_cost = np.sqrt(arrangement_cost_vector[0] ** 2 + arrangement_cost_vector[1] ** 2)

    # Calculate the target neighbour cost
    for curr_agent_name, target_relative_position in target_neighbours.items():
        assert beliefs[int(agent_name)] is not None

        curr_agent_position = beliefs[int(curr_agent_name)]
        agent_position = beliefs[int(agent_name)]

        if curr_agent_position is None:
            continue
        target_neighbor_cost += np.sqrt(
            (curr_agent_position[0] - agent_position[0] - target_relative_position[0])
            ** 2
            + (curr_agent_position[1] - agent_position[1] - target_relative_position[1])
            ** 2
        )

    # Return weighted cost
    return (config.global_reward_weight * arrangement_cost) + (
        config.local_reward_weight * target_neighbor_cost
    )

def step(agent_name, agent_instance, observations, infos, env, config):
  
    """
    This function is called every step of the simulation. It is responsible for
    calculating the cost for every possible action and choosing the action with
    the lowest cost. It also updates the agent's beliefs with the new information
    it has received.
    """

    # Calculate the cost for every possible action
    action_costs = {}
    for action in range(env.action_space(agent_name).n):
        # Calculate the cost for the action
        observations[agent_name] = (
            agent_instance.p_pos[0] + config.possible_moves[action][0],
            agent_instance.p_pos[1] + config.possible_moves[action][1],
        )
        action_costs[action] = calculate_cost(
            agent_name,
            agent_instance.target_neighbour,
            observations,
            config.size,
            config,
        )

        # Reset the original position of the agent
        observations[agent_name] = (
            agent_instance.p_pos[0],
            agent_instance.p_pos[1],
        )

    # Choose the action with the lowest cost
    min_action = min(action_costs, key=action_costs.get)

    return min_action

def main(included_data=None):
    if included_data is None:
        included_data = ["observations", "rewards", "terminations", "truncations", "infos"]

    env_config = BaselineConfig()
    env = nepiada.parallel_env(config=env_config)
    observations, infos = env.reset()

    results = []  # List to store results

    agents = env.agents

    while env.agents:
        actions = {}
        for curr_agent_name in env.agents:
            curr_agent_instance = infos[curr_agent_name]["agent_instance"]
            agent_action = step(
                int(curr_agent_name),
                curr_agent_instance,
                observations[curr_agent_name]["beliefs"],
                env,
                env_config,
            )
            actions[curr_agent_name] = agent_action

        observations, rewards, terminations, truncations, infos = env.step(actions)
        step_result = {}
        if "observations" in included_data:
            step_result['observations'] = observations
        if "rewards" in included_data:
            step_result['rewards'] = rewards
        if "terminations" in included_data:
            step_result['terminations'] = terminations
        if "truncations" in included_data:
            step_result['truncations'] = truncations
        if "infos" in included_data:
            step_result['infos'] = infos

        # Store relevant information in results
        results.append(step_result)

    env.close()
    pygame.quit()

    return results, agents, env_config, env  # Return the collected results

if __name__ == "__main__":
    main()