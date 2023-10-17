import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class DroneEnv(gym.Env):
    def __init__(self, D=5, max_pos=10, max_steps=50):
        super(DroneEnv, self).__init__()

        self.D = D
        self.max_pos = max_pos
        self.max_steps = max_steps
        self.current_step = 0

        self.drone1_pos = np.random.randint(-self.max_pos, self.max_pos)
        self.drone2_pos = np.random.randint(-self.max_pos, self.max_pos)

        self.action_space = spaces.MultiDiscrete([3, 3])  # -1, 0, 1 for each drone
        self.observation_space = spaces.Box(low=-self.max_pos, high=self.max_pos, shape=(2,), dtype=np.int32)

        self.positions_history = []

    def reset(self):
        self.drone1_pos = np.random.randint(-self.max_pos, self.max_pos)
        self.drone2_pos = np.random.randint(-self.max_pos, self.max_pos)
        self.current_step = 0
        self.positions_history = [(self.drone1_pos, self.drone2_pos)]
        return np.array([self.drone1_pos, self.drone2_pos])

    def step(self, action):
        self.current_step += 1

        actions = [-1, 0, 1]
        action_drone1 = actions[action[0]]
        action_drone2 = actions[action[1]]

        self.drone1_pos = np.clip(self.drone1_pos + action_drone1, -self.max_pos, self.max_pos)
        self.drone2_pos = np.clip(self.drone2_pos + action_drone2, -self.max_pos, self.max_pos)

        self.positions_history.append((self.drone1_pos, self.drone2_pos))

        reward = - (self.drone1_pos**2 + self.drone2_pos**2 + (abs(self.drone1_pos - self.drone2_pos) - self.D)**2)
        done = self.current_step >= self.max_steps or (action_drone1 == 0 and action_drone2 == 0)
        print(f"Iteration {self.current_step}: Drone1: {self.drone1_pos}, Drone2: {self.drone2_pos}")


        return np.array([self.drone1_pos, self.drone2_pos]), reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class QLearning:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_epsilon=0.9):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_epsilon

        max_pos = env.max_pos
        self.q_table = np.zeros((2*max_pos+1, 2*max_pos+1, env.action_space.nvec[0], env.action_space.nvec[1]))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.unravel_index(self.q_table[state[0]+self.env.max_pos, state[1]+self.env.max_pos].argmax(), 
                                    self.q_table[state[0]+self.env.max_pos, state[1]+self.env.max_pos].shape)

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                predict = self.q_table[state[0]+self.env.max_pos, state[1]+self.env.max_pos, action[0], action[1]]
                target = reward + self.gamma * np.max(self.q_table[next_state[0]+self.env.max_pos, next_state[1]+self.env.max_pos])
                self.q_table[state[0]+self.env.max_pos, state[1]+self.env.max_pos, action[0], action[1]] += self.lr * (target - predict)
                
                state = next_state

            if self.epsilon > 0.05:
                self.epsilon *= 0.995


def visualize_drones_movement(history):
    plt.figure(figsize=(12, 6))

    drone1_positions = [pos[0] for pos in history]
    drone2_positions = [pos[1] for pos in history]
    steps = list(range(len(history)))

    plt.plot(steps, drone1_positions, '-ro', label='Drone 1', markersize=5)
    plt.plot(steps, drone2_positions, '-bo', label='Drone 2', markersize=5)
    
    plt.title('Drones Movement Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    env = DroneEnv(D=5, max_pos=10, max_steps=50)
    agent = QLearning(env)
    agent.train(episodes=500)

    visualize_drones_movement(env.positions_history)
