import ray
import supersuit as ss
from ray import tune, air, train
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

import env.nepiada as nepiada
from utils.config import Config


def env_creator(args):
    nepiada_config = Config()
    env = nepiada.parallel_env(config=nepiada_config)
    # env = ss.dtype_v0(env, "float32")
    return ParallelPettingZooEnv(env)


if __name__ == "__main__":
    ray.init()

    env = env_creator({})
    register_env("nepiada", env_creator)

    config = DQNConfig()

    replay_config = {
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": 60000,
            "prioritized_replay_alpha": 0.5,
            "prioritized_replay_beta": 0.5,
            "prioritized_replay_eps": 3e-6,
        }

    config = config.training(replay_buffer_config=replay_config, num_atoms=51, n_step=3, noisy=True, v_min=-10.0, v_max=10.0, gamma=0.99, lr=0.0001, train_batch_size=32)
    config = config.resources(num_gpus=1)
    config = config.rollouts(num_rollout_workers=0, rollout_fragment_length=4, compress_observations=True)
    config = config.environment("nepiada")
    config = config.exploration(explore=True, exploration_config={"type": "EpsilonGreedy", "initial_epsilon": 1.0, "final_epsilon": 0.01, "epsilon_timesteps": 200000})
    config = config.multi_agent(policies=env.get_agent_ids(), policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id))

    # algo = DQN(config=config)
    # algo.train()

    # stop = {"episodes_total": 100000}
    # tune.Tuner(
    #     "DQN",
    #     run_config=air.RunConfig(stop=stop, checkpoint_config=train.CheckpointConfig(checkpoint_frequency=50)),
    #     param_space=config
    # ).fit()
    # tuner = tune.Tuner.restore(
    #     path="C:/Users/thano/ray_results/DQN_2024-02-09_21-41-17",
    #     trainable="DQN",
    #     param_space=config,
    #     resume_errored=True
    # )
    # tuner.fit()
    # test_config = DQNConfig()
    # test_config = test_config.rollouts(num_rollout_workers=0, rollout_fragment_length=4, compress_observations=True)
    # test_config = test_config.environment("nepiada")
    # test_config = test_config.multi_agent(policies=env.get_agent_ids(), policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id))

    # algo = DQN(config=config)
    # algo.restore("C:/Users/thano/ray_results/DQN_2024-02-10_04-56-55/DQN_nepiada_b8fa7_00000_0_2024-02-10_04-56-55/checkpoint_000019")
    # observations, infos = env.reset()
    # while True:
    #     actions = {}
    #     for curr_agent_name in env.get_agent_ids():
    #         actions[curr_agent_name] = algo.compute_single_action(observation=observations[curr_agent_name], policy_id=curr_agent_name)
    #     print(actions)
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    # env.close()
    