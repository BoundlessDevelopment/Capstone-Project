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
            "capacity": 50000,
            "prioritized_replay_alpha": 0.5,
            "prioritized_replay_beta": 0.5,
            "prioritized_replay_eps": 3e-6,
        }

    config = config.training(replay_buffer_config=replay_config, num_atoms=51, n_step=3, noisy=True, v_min=-10.0, v_max=10.0, gamma=0.99, lr=0.0001, train_batch_size=32)
    config = config.resources(num_gpus=1)
    config = config.rollouts(num_rollout_workers=11, rollout_fragment_length=4, compress_observations=True)
    config = config.environment("nepiada")
    config = config.exploration(explore=True, exploration_config={"type": "EpsilonGreedy", "initial_epsilon": 1.0, "final_epsilon": 0.01, "epsilon_timesteps": 10000})

    # algo = DQN(config=config)
    # algo.train()

    # stop = {"episodes_total": 60000}
    # tune.Tuner(
    #     "DQN",
    #     run_config=air.RunConfig(stop=stop, checkpoint_config=train.CheckpointConfig(checkpoint_frequency=50)),
    #     param_space=config
    # ).fit()

    # test_config = DQNConfig()
    # test_config = test_config.rollouts(num_rollout_workers=0)
    # test_config = test_config.environment("nepiada")
    # test_config = test_config.multi_agent(policies=env.get_agent_ids(), policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id))

    # algo = DQN(config=test_config)
    # algo.restore("C:/Users/thanos/results/DQN_2024-02-02_14-29-18/DQN_nepiada_5c7b1_00000_0_num_atoms=1_2024-02-02_14-29-19/checkpoint_000005")
    # env.reset()
    # observations, infos = env.reset()
    # while True:
    #     actions = {}
    #     for curr_agent_name in env.get_agent_ids():
    #         actions[curr_agent_name] = algo.compute_single_action(observation=observations[curr_agent_name], policy_id=curr_agent_name)
    #     print(actions)
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    # env.close()
    