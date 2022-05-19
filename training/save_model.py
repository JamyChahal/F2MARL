import argparse
import json
import pickle
from copy import deepcopy
from enum import Enum

from numpy import float32
import os

from ray.rllib.models import ModelCatalog

from behaviors.reactive_walking_policy import ReactiveWalkingPolicy
from environment.mpe import obs_patrol_v0

import ray
from ray.tune.trial import ExportFormat
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from behaviors.random_walking_policy import RandomWalkingPolicy
from training.custom_models.centralizedCriticModel import ObservationCentralizedCriticModel
from training.custom_models.keras_nn_model import NNKerasModel

from training.train import get_config, get_trainer_from_backup, get_trainer_class, get_saved_frequency, \
    save_model, env_creator, METHOD


def main(args):
    with open(args.params) as json_param:
        params = json.load(json_param)
    # Import centralized critic model
    ModelCatalog.register_custom_model("cc_model", ObservationCentralizedCriticModel)

    reward = params["reward"]  # ind, col etc.
    method = METHOD[params["model"]["type"]]

    alg_name = "PPO"

    config = deepcopy(get_trainer_class(alg_name)._default_config)
    register_env('obs_patrol', lambda config: env_creator(params))

    test_env = env_creator(params)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    policies_agent = {"policy_agent": (None, obs_space, act_space, {})}
    policies_target = {"policy_target_{}".format(i): (ReactiveWalkingPolicy, obs_space, act_space, {})
                       for i in range(params['nbr_targets'])}

    policies = {**policies_agent, **policies_target}  # Merge both policies
    policy_ids = list(policies.keys())

    # https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_two_trainers.py
    def policy_mapping_fn(agent_id, episode, **kwargs):
        if agent_id.startswith("agent_"):  # We have a single policy for agent
            return 'policy_agent'
        else:  # But each target has one
            id = agent_id.split("adversary_", 1)[1]
            return 'policy_target_' + id

    config.update({
        "env": "obs_patrol",
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["policy_agent"],  # We train only agent, not target
        },
        "model": {
            "custom_model": "cc_model",
            "fcnet_hiddens": params['model']['fcnet_hiddens'],
            "fcnet_activation": "tanh",
        },
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "log_level": "DEBUG",
        "num_workers": params['num_workers'],  # 6
    })
    #config = get_config(method, config, params)

    #####################
    ray.init()

    trainer = get_trainer_class(alg_name)(env="obs_patrol", config=config)
    trainer = get_trainer_from_backup(trainer, method, reward)

    save_model(trainer, method, reward)

    test_env.reset()
    ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    params_default = "params.list"
    parser.add_argument("-p", "--params", type=str, default=params_default)
    args = parser.parse_args()
    main(args)
