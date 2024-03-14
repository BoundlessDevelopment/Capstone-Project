import ray
import supersuit as ss
from ray import tune, air, train
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.policy.policy import Policy
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv, PettingZooEnv
from ray.tune.registry import register_env
from pettingzoo.utils.conversions import parallel_to_aec

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

    config = config.training(replay_buffer_config=replay_config, num_atoms=51, n_step=3, noisy=True, v_min=-10.0, v_max=10.0, gamma=0.99, lr=0.00025, train_batch_size=64, double_q=True)
    config = config.resources(num_gpus=1)
    config = config.rollouts(num_rollout_workers=0, rollout_fragment_length=4, compress_observations=True)
    config = config.environment("nepiada")
    config = config.exploration(explore=False, exploration_config={"type": "EpsilonGreedy", "initial_epsilon": 1.0, "final_epsilon": 0.00, "epsilon_timesteps": 1000})
    config = config.multi_agent(policies=env.get_agent_ids(), policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id))
    
    ### TRAINING ####
    # algo = DQN(config=config)

    # for i in range(10000):
    #     result = algo.train()
    #     print("train iteration: ", str(i))
    #     if result["training_iteration"] % 50 == 0:
    #         checkpoint = algo.save(checkpoint_dir="C:/Users/thano/ray_results/Rainbow_Feb11_Night/" + str(i) + "_" + str(result["episode_reward_mean"]))
    #         print("checkpoint saved at: ", checkpoint.checkpoint.path)
    #         print(str(result))

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


    #### EVALUATION #####
    # algo = DQN(config=config)
    # algo.restore("C:/Users/thano/ray_results/Rainbow_Feb11_Night/4049_156.08894115214034")
    
    # test_env_config = Config()
    # test_env = nepiada.parallel_env(config=test_env_config)

    # observations, infos = test_env.reset()
    # while test_env.agents:
    #     actions = {}
    #     for curr_agent_name in test_env.agents:
    #         actions[curr_agent_name] = algo.compute_single_action(observation=observations[curr_agent_name], policy_id=curr_agent_name)
    #         print(actions)
    #     print(actions)
    #     observations, rewards, terminations, truncations, infos = test_env.step(actions)
    #     test_env.render()
    # test_env.close()

#    rl_policy = Policy.from_checkpoint("C:/Users/thano/ray_results/Rainbow_Feb11_Night/4049_156.08894115214034")

    # observations, infos = test_env.reset()
    # while test_env.agents:
    #     actions = {}
    #     for curr_agent_name in test_env.agents:
    #         actions[curr_agent_name] = rl_policy[curr_agent_name].compute_single_action(observation=observations[curr_agent_name])
    #         print(actions)
    #     print(actions)
    #     observations, rewards, terminations, truncations, infos = test_env.step(actions)
    #     test_env.render()
    # test_env.close()

    