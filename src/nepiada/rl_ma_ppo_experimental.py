import ray
import supersuit as ss
from ray import tune, air, train
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv, PettingZooEnv
from ray.tune.registry import register_env
from pettingzoo.utils import parallel_to_aec

import env.nepiada as nepiada
from utils.config import Config


def env_creator(args):
    nepiada_config = Config()
    env = nepiada.parallel_env(config=nepiada_config)
    # env = ss.dtype_v0(env, "float32")
    # IGNORE THIS - TESTING PURPOSES
    # return parallel_to_aec(env)
    return env


if __name__ == "__main__":    
    env = env_creator()
    register_env("nepiada", lambda config: ParallelPettingZooEnv(env_creator(config)))
    ray.init()
    
    config = PPOConfig()
    config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size=128)
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=0)
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
    
    # algo = config.build(env="nepiada")
    # algo.restore("C:/Users/thanos/testing/PPO_2024-02-01_22-57-05/PPO_nepiada_21646_00000_0_2024-02-01_22-57-06/checkpoint_000486")
    # env.reset()

    # for agent in env.agent_iter():
    #     observation, reward, termination, truncation, info = env.last()
    #     action = algo.compute_single_action(observation)

    #     env.step(action)
    # env.close()