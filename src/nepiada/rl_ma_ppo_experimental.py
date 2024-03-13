import ray
import supersuit as ss
from ray import tune, air, train
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv, PettingZooEnv
from ray.tune.registry import register_env
from pettingzoo.utils import parallel_to_aec

import env.nepiada as nepiada
from utils.config import Config

def env_creator(args):
    nepiada_config = Config()
    env = nepiada.parallel_env(config=nepiada_config)
    # env = ss.dtype_v0(env, "float32")
    return ParallelPettingZooEnv(env)


if __name__ == "__main__":    
  #  ray.init()

 #   env = env_creator({})
    register_env("nepiada", env_creator)
    
    config = PPOConfig()
    config = config.training(gamma=0.67, lr=0.00005, kl_coeff=0.3, train_batch_size=128, vf_clip_param=100, clip_param=0.2, entropy_coeff=0.001)
    config = config.resources(num_gpus=1)
    config = config.rollouts(num_rollout_workers=0)
    config = config.multi_agent(policies=["parameter_sharing"], policy_mapping_fn=(lambda agent_id, *args, **kwargs: "parameter_sharing"))
    config = config.environment("nepiada")


    algo = config.build()

    ## TRAINING #####
    # best_reward_mean = -100000000
    # iterations_since_last_checkpoint = 0
    # name_of_experiment = "PPO_Mar12_ParamShare_LocalReward_Gamma067_LR00001_3A3T_2"
    # algo.restore("C:/Users/thano/ray_results/PPO_Mar12_ParamShare_LocalReward_Gamma067_LR00001_3A3T_2/5224_-30569.544658786097")
    # for i in range(100000):
    #     result = algo.train()
    #     print(f"Training iteration: {str(i)} | Reward mean: {str(result['episode_reward_mean'])}")
    #     iterations_since_last_checkpoint += 1
    #     if result["episode_reward_mean"] > best_reward_mean:
    #         print(f"New best reward mean: {str(result['episode_reward_mean'])} | Previous best: {str(best_reward_mean)}")
    #         best_reward_mean = result["episode_reward_mean"]
    #         checkpoint = algo.save(checkpoint_dir="C:/Users/thano/ray_results/" + name_of_experiment + "/" + str(i) + "_" + str(result["episode_reward_mean"]))
    #         print("Checkpoint saved at: ", checkpoint.checkpoint.path)
    #         iterations_since_last_checkpoint = 0
    #     elif iterations_since_last_checkpoint > 50:
    #         print("Iterations since last checkpoint exceeded threshold | Saving checkpoint...")
    #         checkpoint = algo.save(checkpoint_dir="C:/Users/thano/ray_results/" + name_of_experiment + "/" + str(i) + "_TOCHECK_" + str(result["episode_reward_mean"]))
    #         print("Checkpoint saved at: ", checkpoint.checkpoint.path)
    #         iterations_since_last_checkpoint = 0

    # stop = {"episodes_total": 60000}

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

    ### EVALUATION ####

    # algo.restore("C:/Users/thano/ray_results/PPO_Mar12_ParamShare_LocalReward_Gamma067_LR00001_3A3T_2/5224_-30569.544658786097")
    
    # test_env_config = Config()
    # test_env = nepiada.parallel_env(config=test_env_config)

    # observations, infos = test_env.reset()
    # while test_env.agents:
    #     actions = {}
    #     for curr_agent_name in test_env.agents:
    #         actions[curr_agent_name] = algo.compute_single_action(observation=observations[curr_agent_name], policy_id="parameter_sharing")
    #     print(actions)
    #     observations, rewards, terminations, truncations, infos = test_env.step(actions)
    #     test_env.render()
    # test_env.close()