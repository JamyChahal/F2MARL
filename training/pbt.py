#!/usr/bin/env python
"""Example of using PBT with RLlib.
Note that this requires a cluster with at least 8 GPUs in order for all trials
to run concurrently, otherwise PBT will round-robin train the trials which
is less efficient (or you can set {"gpu": 0} to use CPUs for SGD instead).
Note that Tune in general does not need 8 GPUs, and this is just a more
computationally demanding example.
Based on https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/pbt_ppo_example.py
"""
import argparse
import json
import os
import random
from copy import deepcopy

from ray.tune.trial import ExportFormat

from behaviors.random_walking_policy import make_randomBehavior
from behaviors.reactive_walking_policy import make_reactiveBehavior
from ray import tune
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.env import PettingZooEnv
from ray.tune import register_env
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.models import ModelCatalog
from environment.mpe import obs_patrol_v0
from training.custom_models.centralizedCriticModel import ObservationCentralizedCriticModel


def env_creator(params):
    reward = params['reward']
    is_reward_individual = True
    is_reward_shared = False
    if reward == 'ind':
        is_reward_individual = True
        is_reward_shared = False
    elif reward == 'col':
        is_reward_individual = False
        is_reward_shared = False
    elif reward == 'col_shared':
        is_reward_individual = False
        is_reward_shared = True
    else:
        print("ERROR : Wrong reward description when creating environment")
        exit()
    env = obs_patrol_v0.env(nbr_agent=params['nbr_agents'], nbr_target=params['nbr_targets'],
                            obs_range=params['obs_range'], safety_range=params['safety_range'],
                            dangerous_range=params['dangerous_range'], com_range=params['com_range'],
                            map_size=params['map_size'], obs_to_normalize=True, max_cycles=params["max_cycles"],
                            is_reward_individual=is_reward_individual, gradual_reward=False, has_protection_force=True,
                            is_reward_shared=is_reward_shared)
    env.reset()
    # print(env.observation_spaces.values())
    env = PettingZooEnv(env)
    return env


def main(args):
    with open(args.params) as json_param:
        params = json.load(json_param)

    alg_name = "PPO"
    dir_name = args.name
    to_resume = os.path.isdir(os.path.expanduser('~') + "/ray_results/" + dir_name)

    # Import centralized critic model
    ModelCatalog.register_custom_model("cc_model", ObservationCentralizedCriticModel)

    config = deepcopy(get_trainer_class(alg_name)._default_config)
    register_env('obs_patrol', lambda config: env_creator(params))

    # register_env('obs_patrol', lambda PettingZoo: env_creator(params))

    test_env = env_creator(params)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    if params['target_behavior'] == 'random':
        target_behavior = make_randomBehavior(params["map_size"])
    elif params['target_behavior'] == 'evasive':
        target_behavior = make_reactiveBehavior(params["map_size"], params["obs_range"])
    else:
        print("ERROR : Target behavior not specified")
        exit()

    policies_agent = {"default_policy": (None, obs_space, act_space, {})}
    policies_target = {"policy_target_{}".format(i): (target_behavior, obs_space, act_space, {}) for i in
                       range(params['nbr_targets'])}

    policies = {**policies_agent, **policies_target}  # Merge both policies

    def policy_mapping_fn(agent_id, episode, **kwargs):
        if agent_id.startswith("agent_"):  # We have a single policy for agent
            return 'default_policy'
        else:  # But each target has one
            id = agent_id.split("adversary_", 1)[1]
            return 'policy_target_' + id

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "entropy_coeff": lambda: random.uniform(0, 0.1),
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 512),  # 128, 16384
            "train_batch_size": lambda: random.randint(2000, 5000),  # 160000
        },
        custom_explore_fn=explore)

    # Update the config dict
    config.update({
        "env": "obs_patrol",
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["default_policy"],  # We train only agent, not target
        },
        "log_level": "DEBUG",
        "kl_coeff": 1.0,
        "num_workers": 6,
        "num_gpus": 0,
        "model": {
            "custom_model": "cc_model",
            "fcnet_hiddens": params['model']['fcnet_hiddens']
        },
        # These params are tuned from a fixed starting value.
        "lambda": 0.95,
        "clip_param": 0.2,
        "lr": 1e-4,
        # These params start off randomly drawn from a set.
        "num_sgd_iter": tune.choice([10, 20, 30]),
        "sgd_minibatch_size": tune.choice([128, 512, 2048]),
        "train_batch_size": tune.choice([10000, 20000, 40000])
        # "train_batch_size": tune.choice([100, 200, 400])
    })

    analysis = tune.run(
        "PPO",
        name=dir_name,
        scheduler=pbt,
        num_samples=6,  # 6
        metric="episode_reward_mean",
        mode="max",
        config=config,
        export_formats=[ExportFormat.MODEL],
        resume=to_resume,
        checkpoint_freq=50,
        checkpoint_at_end=True,
        keep_checkpoints_num=50,
        # stop={"training_iteration": 2000}
        stop={"episode_reward_mean": 7.0}
    )

    print("best hyperparameters: ", analysis.best_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    params_default = "params_pbt.list"
    name_default = "pbt_test_f10"
    parser.add_argument("-p", "--params", type=str, default=params_default)
    parser.add_argument("-n", "--name", type=str, default=name_default)
    args = parser.parse_args()
    main(args)
