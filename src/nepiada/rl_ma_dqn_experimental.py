import ray
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch
import supersuit as ss
from ray import tune, air, train
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

import env.nepiada as nepiada
from utils.config import Config

class NepiadaCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
    
    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs) -> None:
        # Get current epsilon for all agents
        agent_epsilons = ""
        for curr_agent_name in worker.env.get_agent_ids():
            curr_epsilon = worker.policy_map[curr_agent_name].get_exploration_state()["cur_epsilon"]
            curr_timestep = worker.policy_map[curr_agent_name].get_exploration_state()["last_timestep"]
            agent_epsilons += f"{curr_agent_name}: {str(curr_epsilon)} | {str(curr_timestep)} || "
        # curr_epsilon = worker.policy_map["parameter_sharing"].get_exploration_state()["cur_epsilon"]
        # curr_timestep = worker.policy_map["parameter_sharing"].get_exploration_state()["last_timestep"]
        # agent_epsilons += f"{str(curr_epsilon)} | {str(curr_timestep)} || "
        print(f"DQN Rollout | Epsilons: {agent_epsilons}")

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

    config = config.training(replay_buffer_config=replay_config, num_atoms=1, gamma=0.5, lr=0.0005)
    config = config.resources(num_gpus=1)
    config = config.rollouts(num_rollout_workers=0, rollout_fragment_length=100, compress_observations=True)
    config = config.environment("nepiada")
    config = config.multi_agent(policies=env.get_agent_ids(), policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id))
    config = config.exploration(explore=True, exploration_config={"type": "EpsilonGreedy", "initial_epsilon": 1.0, "final_epsilon": 0.01, "epsilon_timesteps": 1000000})
    config = config.callbacks(NepiadaCallbacks)

    ### TRAINING #####
    algo = DQN(config=config)
    best_reward_mean = -100000000
    iterations_since_last_checkpoint = 0
    name_of_experiment = "DQN_Mar15_MA_LocalRewardDELTA_TARNORM_LR00005_Gamma05_6A3T_1"
    for i in range(100000):
        result = algo.train()
        print(f"Training iteration: {str(i)} | Reward mean: {str(result['episode_reward_mean'])}")
        iterations_since_last_checkpoint += 1
        if result["episode_reward_mean"] > best_reward_mean:
            print(f"New best reward mean: {str(result['episode_reward_mean'])} | Previous best: {str(best_reward_mean)}")
            best_reward_mean = result["episode_reward_mean"]
            checkpoint = algo.save(checkpoint_dir="C:/Users/thano/ray_results/" + name_of_experiment + "/" + str(i) + "_" + str(result["episode_reward_mean"]))
            print("Checkpoint saved at: ", checkpoint.checkpoint.path)
            iterations_since_last_checkpoint = 0
        elif iterations_since_last_checkpoint > 50:
            print("Iterations since last checkpoint exceeded threshold | Saving checkpoint...")
            checkpoint = algo.save(checkpoint_dir="C:/Users/thano/ray_results/" + name_of_experiment + "/" + str(i) + "_TOCHECK_" + str(result["episode_reward_mean"]))
            print("Checkpoint saved at: ", checkpoint.checkpoint.path)
            iterations_since_last_checkpoint = 0

    stop = {"episodes_total": 60000}
    tune.Tuner(
        "DQN",
        run_config=air.RunConfig(stop=stop, checkpoint_config=train.CheckpointConfig(checkpoint_frequency=50)),
        param_space=config
    ).fit()

    ### EVALUATION ####

    # algo = DQN(config=config)
    # algo.restore("C:/Users/thano/ray_results/DQN_Mar14_MA_LocalRewardDELTA_TARNORM_LR00005_Gamma05_4A2T_1/4542_199.03627875204452")
    
    # test_env_config = Config()
    # test_env = nepiada.parallel_env(config=test_env_config)

    # observations, infos = test_env.reset()
    # while test_env.agents:
    #     actions = {}
    #     for curr_agent_name in test_env.agents:
    #         actions[curr_agent_name] = algo.compute_single_action(observation=observations[curr_agent_name], policy_id=curr_agent_name)
    #     # print(actions)
    #     observations, rewards, terminations, truncations, infos = test_env.step(actions)
    #     test_env.render()
    # test_env.close()

    # algo = DQN(config=test_config)
    # algo.restore("C:/Users/thano/ray_results/DQN_2024-02-02_14-29-18/DQN_nepiada_5c7b1_00000_0_num_atoms=1_2024-02-02_14-29-19/checkpoint_000005")
    # env.reset()
    # observations, infos = env.reset()
    # while True:
    #     actions = {}
    #     for curr_agent_name in env.get_agent_ids():
    #         actions[curr_agent_name] = algo.compute_single_action(observation=observations[curr_agent_name], policy_id=curr_agent_name)
    #     print(actions)
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    # env.close()
    