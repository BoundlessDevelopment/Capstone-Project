from pettingzoo.mpe import simple_adversary_v3
import time

env = simple_adversary_v3.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()
        # Adding a bit of delay to see the frame
        time.sleep(0.1)

    env.step(action)

env.close()