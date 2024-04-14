# Local imports
import numpy as np
import sys
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
        assert beliefs[agent_name] is not None

        # FIX: We need an integer to index beliefs so we prune it from the agent name
        curr_agent_name = int(curr_agent_name[-1:])

        curr_agent_position = beliefs[curr_agent_name]
        agent_position = beliefs[agent_name]

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

def step(agent_name, agent_instance, beliefs, env, config):
  
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
        beliefs[agent_name] = np.array((
            agent_instance.p_pos[0] + config.possible_moves[action][0],
            agent_instance.p_pos[1] + config.possible_moves[action][1],
        ), dtype=np.float32)

        action_costs[action] = calculate_cost(
            agent_name,
            agent_instance.target_neighbour,
            beliefs,
            config.size,
            config,
        )

        # Reset the original position of the agent
        beliefs[agent_name] = agent_instance.p_pos

    # Choose the action with the lowest cost
    min_action = min(action_costs, key=action_costs.get)
    return min_action

def main(seed, truthful, adversarial, width, height, radius, noise_type, iterations, included_data=None):
    if included_data is None:
        included_data = ["observations", "rewards", "terminations", "truncations", "infos"]

    env_config = BaselineConfig()
    env_config.set_seed(seed)
    env_config.set_agents(truthful, adversarial, width, height)
    env_config.set_observation_radius(radius)
    env_config.set_noise(noise_type)
    env_config.set_iterations(iterations)

    env = nepiada.parallel_env(config=env_config)
    observations, infos = env.reset()

    results = []  # List to store results

    agents = env.agents

    while env.agents:
        actions = {}
        for curr_agent_name in env.agents:
            curr_agent_instance = infos[curr_agent_name]["agent_instance"]

            agent_action = step(
                # FIX: We need an integer to index beliefs so we prune it from the agent name
                int(curr_agent_name[-1:]),
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

def get_convergence_score(agent_list, env_config):
    # Get the global convergence score
    scores = {}
    target_x = env_config.size / 2
    target_y = env_config.size / 2
    global_arrangement_vector = np.array([0.0, 0.0])
    for agent_name, agent_instance in agent_list.items():
        agent = agent_instance["agent_instance"]
        global_arrangement_vector += np.array(
            [(agent.p_pos[0] - target_x), (target_y - agent.p_pos[1])]
        )

    global_arrangement_vector = np.divide(
        global_arrangement_vector, len(agent_list)
    )
    global_arrangement_reward = np.sqrt(
        global_arrangement_vector[0] ** 2 + global_arrangement_vector[1] ** 2
    )

    # Get the local convergence score
    for agent_name, agent_instance in agent_list.items():
        agent = agent_instance["agent_instance"]
        agent_x = agent.p_pos[0]
        agent_y = agent.p_pos[1]
        deviation_from_arrangement = 0
        for neighbour_name, ideal_distance in agent.target_neighbour.items():
            neighbour = agent_list[neighbour_name]["agent_instance"]
            neighbour_x = neighbour.p_pos[0]
            neighbour_y = neighbour.p_pos[1]
            ideal_x = ideal_distance[0]
            ideal_y = ideal_distance[1]

            deviation_from_arrangement += np.sqrt(
                (neighbour_x - agent_x - ideal_x) ** 2
                + (neighbour_y - agent_y - ideal_y) ** 2
            )

        scores[agent_name] = -(env_config.local_reward_weight * deviation_from_arrangement)

    local_convergence_score = sum(scores.values()) / len(scores.values())
    global_convergence_score = -(env_config.global_reward_weight * global_arrangement_reward)

    return local_convergence_score, global_convergence_score

if __name__ == "__main__":
    # Get arguments from command line
    seed = int(sys.argv[1])
    truthful = int(sys.argv[2])
    adversarial = int(sys.argv[3])
    width = int(sys.argv[4])
    height = int(sys.argv[5])
    radius = int(sys.argv[6])
    noise_type = sys.argv[7]
    iterations = int(sys.argv[8])

    results, agents, env_config, env = main(seed, truthful, adversarial, width, height, radius, noise_type, iterations)

    local_convergence_score, global_convergence_score = get_convergence_score(results[-1]["infos"], env_config)
    print(local_convergence_score, global_convergence_score)