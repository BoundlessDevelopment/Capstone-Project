import ray
import supersuit as ss
from ray import tune, air, train
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

import env.nepiada as nepiada
from utils.config import Config


def env_creator(args):
    nepiada_config = Config()
    env = nepiada.parallel_env(config=nepiada_config)
    # env = ss.dtype_v0(env, "float32")
    return env


if __name__ == "__main__":
    ray.init()

   # register_env("nepiada", lambda config: ParallelPettingZooEnv(env_creator(config)))
    register_env("nepiada", lambda config: ParallelPettingZooEnv(env_creator(config)))

    config = DQNConfig()

    # replay_config = {
    #         "type": "MultiAgentPrioritizedReplayBuffer",
    #         "capacity": 60000,
    #         "prioritized_replay_alpha": 0.5,
    #         "prioritized_replay_beta": 0.5,
    #         "prioritized_replay_eps": 3e-6,
    #     }

    # config = config.training(replay_buffer_config=replay_config, num_atoms=tune.grid_search([1,]))
    # config = config.resources(num_gpus=1)
    # config = config.rollouts(num_rollout_workers=11)
    # config = config.environment("nepiada")
    # # algo = DQN(config=config)
    # # algo.train()
    # tune.Tuner(
    #     "DQN",
    #     run_config=air.RunConfig(stop={"training_iteration":175}, checkpoint_config=train.CheckpointConfig(checkpoint_frequency=30)),
    #     param_space=config.to_dict()
    # ).fit()

    # DQN_agent = DQN.from_checkpoint("C:/Users/thanos/RL_experimental/DQN_2024-02-01_21-40-46/DQN_nepiada_77c1a_00000_0_num_atoms=1_2024-02-01_21-40-46/checkpoint_000005")
    # env = env_creator(None)
    # env.reset()

    # while env.agents:
    #     observations, rewards, terminations, truncations, infos = env.last()
    #     actions = DQN_agent.compute_actions(observations)

    #     env.step(actions)
    # env.close()