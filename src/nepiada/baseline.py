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
    for curr_agent_name, agent_belief in beliefs.items():
        if agent_belief is None:
            length -= 1
        else:
            arrangement_cost_vector += np.array([(agent_belief[0] - target_x), (target_y - agent_belief[1])])

    # Normalize the global arrangement_cost_vector
    if (length != 0):
        arrangement_cost_vector = np.divide(arrangement_cost_vector, length)

    arrangement_cost = np.sqrt(arrangement_cost_vector[0] ** 2 + arrangement_cost_vector[1] ** 2)

    # Calculate the target neighbour cost
    for curr_agent_name, target_relative_position in target_neighbours.items():
        assert beliefs[agent_name] is not None

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
        if (
            current_agent not in incoming_messages
            or incoming_messages[current_agent] is None
            or len(incoming_messages[current_agent].items()) == 0
        ):
            # No incoming messages about this agent, keep previous state
            new_beliefs[current_agent] = curr_beliefs[current_agent]
            continue

        in_messages = []

        # Get incoming messages that contain this agent's position
        for _, comm_message in incoming_messages[current_agent].items():
            if comm_message is not None:
                in_messages.append(comm_message)

        if curr_beliefs[current_agent] is None:
            # Average all the incoming messages for the case where we don't have an estimate for the current agent
            x_pos_mean = sum([message[0] for message in in_messages]) / len(in_messages)
            y_pos_mean = sum([message[1] for message in in_messages]) / len(in_messages)
            new_beliefs[current_agent] = (x_pos_mean, y_pos_mean)
            continue
            
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

def weighted_beliefs(incoming_messages, new_beliefs, agent_name, all_agents, agent_instance):
    #print(agent_name)
    for target_agent in all_agents:
        positions = []
        for talking_agent in all_agents:
            if target_agent in incoming_messages and \
                    talking_agent in incoming_messages[target_agent]:
                positions.append(incoming_messages[target_agent][talking_agent])
            else:
                positions.append(None)
        finalx, finaly, denom, empty = 0,0,0, True
        for i in range(len(positions)):
            if(positions[i] != None):
                empty = False
                finalx += positions[i][0]*agent_instance.truthful_weights[i]
                finaly += positions[i][1]*agent_instance.truthful_weights[i]
                denom += agent_instance.truthful_weights[i]
        if(not empty):
            new_beliefs[target_agent] = (finalx/denom, finaly/denom)
        elif (agent_instance.beliefs[target_agent] is not None):
            new_beliefs[target_agent] = (agent_instance.beliefs[target_agent][0], agent_instance.beliefs[target_agent][1])
        #print(target_agent)
        #print(positions)

def step(agent_name, agent_instance, observations, infos, env, config):
  
    """
    This function is called every step of the simulation. It is responsible for
    calculating the cost for every possible action and choosing the action with
    the lowest cost. It also updates the agent's beliefs with the new information
    it has received.
    """
    # Create new beliefs dict with observation information
    new_beliefs = create_beliefs_with_obs(agent_name, observations, env.agents)
    # If there are incoming messages, process them and update beliefs
    # Incoming messages should never be None after the first step

    if "incoming_messages" in infos:
        strip_extreme_values_and_update_beliefs(
            config.D,
            infos["incoming_messages"],
            agent_instance.beliefs,
            new_beliefs,
            agent_name,
            env.agents,
        )
        weighted_beliefs(infos["incoming_messages"], new_beliefs, agent_name, env.agents, agent_instance)

    # Calculate the cost for every possible action
    action_costs = {}
    for action in range(env.action_space(agent_name).n):
        # Calculate the cost for the action
        new_beliefs[agent_name] = (
            agent_instance.p_pos[0] + config.possible_moves[action][0],
            agent_instance.p_pos[1] + config.possible_moves[action][1],
        )
        action_costs[action] = calculate_cost(

            agent_name,
            agent_instance.target_neighbour,
            new_beliefs,
            config.size,
            config,
        )

    # Choose the action with the lowest cost
    min_action = min(action_costs, key=action_costs.get)

    new_beliefs[agent_name] = (
        agent_instance.p_pos[0] + config.possible_moves[min_action][0],
        agent_instance.p_pos[1] + config.possible_moves[min_action][1],
    )
    agent_instance.beliefs = new_beliefs

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
                curr_agent_name,
                curr_agent_instance,
                observations[curr_agent_name],
                infos[curr_agent_name],
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
