import nepiada

env = nepiada.parallel_env(render_mode="human", amount_of_agents=5, grid_size=10)
observations, infos = env.reset()

while env.agents:
    # TODO: POLICY
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()