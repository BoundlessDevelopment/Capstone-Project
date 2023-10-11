from pettingzoo.mpe import complex_adversary_v1
import time

env = complex_adversary_v1.env(N=2, render_mode="human")
env.reset(seed=42)
print("This is the new complex adversary")

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