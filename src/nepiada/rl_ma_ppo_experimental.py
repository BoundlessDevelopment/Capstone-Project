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
    return ParallelPettingZooEnv(env)


if __name__ == "__main__":
    ray.init()
    
    env = env_creator({})
    register_env("nepiada", env_creator)

    config = PPOConfig()
    config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size=128)
    config = config.resources(num_gpus=1)
    config = config.rollouts(num_rollout_workers=10)
    config = config.multi_agent(policies=env.get_agent_ids(), policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id))


    # algo = config.build(env="nepiada")
    # algo.train()

    # stop = {"episodes_total": 60000}

    # config = config.environment(env="nepiada")

    # tune.Tuner(
    #     "PPO",
    #     run_config=air.RunConfig(
    #         stop=stop,
    #         checkpoint_config=air.CheckpointConfig(
    #             checkpoint_frequency=30,
    #         ),
    #     ),
    #     param_space=config,
    # ).fit()

    # tune.run(
    #     "PPO",
    #     name="PPO",
    #     stop={"timesteps_total": 50000},
    #     checkpoint_freq=10,
    #     config=config.to_dict(),
    # )
