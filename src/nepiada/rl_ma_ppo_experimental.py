import ray
import supersuit as ss
from ray import tune, air, train
from ray.rllib.algorithms.ppo import PPOConfig
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

    register_env("nepiada", lambda config: ParallelPettingZooEnv(env_creator(config)))

    config = PPOConfig()
    config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size=128)
    config = config.resources(num_gpus=1)
    config = config.rollouts(num_rollout_workers=1)

    # algo = config.build(env="nepiada")
    # algo.train()

    # config = config.environment(env="nepiada")

    # tune.Tuner(
    #     "PPO",
    #     run_config=air.RunConfig(
    #         stop={"training_iteration": 10},
    #         checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
    #     ),
    #     param_space=config.to_dict(),
    # ).fit()

    # tune.run(
    #     "PPO",
    #     name="PPO",
    #     stop={"timesteps_total": 50000},
    #     checkpoint_freq=10,
    #     config=config.to_dict(),
    # )
