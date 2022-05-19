import argparse
import json
import pickle
from copy import deepcopy
from enum import Enum

from numpy import float32
import os

from ray.rllib.models import ModelCatalog

from behaviors.random_walking_policy import make_randomBehavior
from behaviors.reactive_walking_policy import make_reactiveBehavior
from environment.mpe import obs_patrol_v0

import ray
from ray.tune.trial import ExportFormat
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from behaviors.random_walking_policy import RandomWalkingPolicy
from training.custom_models.centralizedCriticModel import ObservationCentralizedCriticModel



class METHOD(Enum):
    NN = 1
    LSTM = 2
    ATTENTION = 3
    CUSTOM_NN = 4
    CUSTOM_LSTM = 5
    CUSTOM_ATTENTION_TRAJECTORY = 6

    @classmethod
    def get_checkpoint_path(cls, method, reward):
        if method == cls.NN:
            return "checkpoint_path_nn_" + reward + ".txt"
        elif method == cls.LSTM:
            return "checkpoint_path_lstm_" + reward + ".txt"
        elif method == cls.CUSTOM_NN:
            return "checkpoint_path_nn_custom_" + reward + ".txt"
        elif method == cls.CUSTOM_LSTM:
            return "checkpoint_path_lstm_custom_" + reward + ".txt"
        elif method == cls.CUSTOM_ATTENTION_TRAJECTORY:
            return "checkpoint_path_att_traj_custom_" + reward + ".txt"
        else:
            print("Method not yet implemented")
            exit()
            return None


def read_last_line(filename):
    with open(filename, 'r') as f:
        last_line = f.readlines()[-1].rstrip('\n')
    return last_line


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
                            map_size=params['map_size'], obs_to_normalize=False, max_cycles=params["max_cycles"],
                            is_reward_individual=is_reward_individual, gradual_reward=False, has_protection_force=True,
                            is_reward_shared=is_reward_shared, share_target=False)
    env.reset()
    print(env.observation_spaces.values())
    env = PettingZooEnv(env)
    return env


def get_config(method, config, params):
    if method == METHOD.ATTENTION:
        print("Not yet implemented")
        exit()
    if method == METHOD.CUSTOM_ATTENTION_TRAJECTORY:
        config.update({
            "model": {
                "custom_model": "frame_stack_model",
                "custom_model_config": {
                    "num_frames": 16,
                }
            },
            "framework": "tf",
        })
    if method == METHOD.CUSTOM_NN:
        config.update({
            "model": {
                "custom_model": "keras_model"
            },
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "log_level": "DEBUG",
            "num_workers": 7,  # 6
            "rollout_fragment_length": 5,  # 30
            "train_batch_size": 20,  # 200
            "sgd_minibatch_size": 20,  # 200
            "horizon": None,  # Terminate after x, by default use done from env
            "no_done_at_end": False,
            # New
            "framework": "tf",  # tf2
            "postprocess_inputs": True,
            "record_env": "videos",  # Not working
            "render_env": False  # Set to True if render every training
        })
    if method == METHOD.NN:
        config.update({
            # "lr": 1e-3,  # Try to reduce lr
            "model": {
                "fcnet_hiddens": params['model']['fcnet_hiddens'],
                "fcnet_activation": "tanh",
            },
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "log_level": "DEBUG",
            "num_workers": 6,  # 6
            "rollout_fragment_length": 30,  # 30
            "train_batch_size": 200,  # 200
            "sgd_minibatch_size": 200,  # 200
            "horizon": None,  # Terminate after x, by default use done from env
            "no_done_at_end": False,
            # New
            "framework": "tf",  # tf2
            "postprocess_inputs": True,
            "record_env": "videos",  # Not working
            "render_env": False  # Set to True if render every training
        })
    if method == METHOD.LSTM:
        config.update({
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "log_level": "DEBUG",
            "num_workers": 8,
            "num_envs_per_worker": 1,
            # "rollout_fragment_length": 30,
            # "train_batch_size": 200,
            # "horizon": 200,  # Terminate after x, by default use done from env
            # "no_done_at_end": False,
            "framework": "tf",
            "vf_share_layers": True,
            "model":
                {
                    "fcnet_hiddens": params['model']['fcnet_hiddens'],
                    "fcnet_activation": "tanh",
                    "use_lstm": True,
                    "lstm_cell_size": params['model']['lstm_cell_size'],  # 256
                    "lstm_use_prev_action": True,
                    "lstm_use_prev_reward": True,
                    # "vf_share_layers": True
                },
            # "vf_loss_coeff": 0.0001,

            # "evaluation_interval": 500,  # Evaluate every x
            # "evaluation_num_episodes": 1,
            # "evaluation_num_workers": 1,
            # "evaluation_config": {
            #    "record_env": "videos",
            #    "render_env": True,  # Set to True if render each evaluation
            # },
            "record_env": "videos",  # Not working, succeed to do it here :
            # https://github.com/ray-project/ray/pull/16428/files/b8a236cde434c0d32182a9cff64a6aaa9795b152
            "render_env": False  # Set to True if render every training
        })
    return config


def get_trainer_from_backup(trainer, method, reward):
    file_name = METHOD.get_checkpoint_path(method, reward)
    if os.path.isfile(file_name):
        checkpoint_path = read_last_line(file_name)
        trainer.restore(checkpoint_path)
        print("Trainer restored from previous training")
        return trainer
    else:
        print("Error : Checkpoint file not found")
        exit()


def save_model(trainer, method, reward):
    if method == METHOD.NN:
        trainer.get_policy("policy_agent").export_model("model_" + reward + "_nn")
    elif method == METHOD.LSTM:
        trainer.get_policy("policy_agent").export_model("model_" + reward + "_lstm")
    elif method == METHOD.CUSTOM_NN:
        trainer.get_policy("policy_agent").export_model("model_" + reward + "_nn_custom")
    else:
        print("Method not recognized, cannot save model, stop here")
        exit()


def get_saved_frequency(method):
    if method == METHOD.NN:
        return 100
    elif method == METHOD.CUSTOM_NN:
        return 50
    elif method == METHOD.LSTM:
        return 20
    else:
        return 10

def main(args):
    with open(args.params) as json_param:
        params = json.load(json_param)

    method = METHOD[params["model"]["type"]]
    train_to_backup = params["train_to_backup"]
    train_from_backup = params["train_from_backup"]
    reward = params["reward"]  # ind, col etc.
    alg_name = "PPO"

    # Import centralized critic model
    ModelCatalog.register_custom_model("cc_model", ObservationCentralizedCriticModel)

    config = deepcopy(get_trainer_class(alg_name)._default_config)
    register_env('obs_patrol', lambda config: env_creator(params))

    test_env = env_creator(params)
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    if params['target_behavior'] == 'random':
        target_behavior = RandomWalkingPolicy
    elif params['target_behavior'] == 'evasive':
        target_behavior = ReactiveWalkingPolicy
    else:
        print("ERROR : Target behavior not specified")
        exit()

    policies_agent = {"policy_agent": (None, obs_space, act_space, {})}
    policies_target = {"policy_target_{}".format(i): (target_behavior, obs_space, act_space, {}) for i in
                       range(params['nbr_targets'])}

    policies = {**policies_agent, **policies_target}  # Merge both policies
    policy_ids = list(policies.keys())

    # https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_two_trainers.py
    def policy_mapping_fn(agent_id, episode, **kwargs):
        if agent_id.startswith("agent_"):  # We have a single policy for agent
            return 'policy_agent'
        else:  # But each target has one
            id = agent_id.split("adversary_", 1)[1]
            return 'policy_target_' + id

    if params['centralized_learning']:
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
                "use_lstm": params['model']['use_lstm'],
                "lstm_cell_size": params['model']['lstm_cell_size']
            },
            "num_gpus": params['num_gpus'],
            "log_level": "DEBUG",
            "num_workers": params['num_workers'],  # 6
            "clip_param": params["clip_param"],
            "lr": params['lr'],
            'entropy_coeff': params['entropy_coeff'],
            "lambda": params['lambda'],
            "num_sgd_iter": params["num_sgd_iter"],
            "sgd_minibatch_size": params['sgd_minibatch_size'],
            "train_batch_size" : params["train_batch_size"]
        })
    else:
        config.update({
            "env": "obs_patrol",
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["policy_agent"],  # We train only agent, not target
            },
            "clip_param": params["clip_param"]
        })
        config = get_config(method, config, params)

    #####################
    ray.init()

    trainer = get_trainer_class(alg_name)(env="obs_patrol", config=config)

    if train_from_backup:
        trainer = get_trainer_from_backup(trainer, method, reward)
    if train_to_backup:
        f = open(METHOD.get_checkpoint_path(method, reward), "a")
        # policy = trainer.get_policy()  # Get the default local policy
        # writer = tf.compat.v1.summary.FileWriter("log_dir", policy.get_session().graph.as_graph_def())

    i = 0
    saved_frequency = get_saved_frequency(method)
    try:
        while True:
            i = i + 1
            print("New training..%s" % i)
            trainer.train()
            if i % saved_frequency == 0 and train_to_backup:
                checkpoint_path = trainer.save()
                f.write(checkpoint_path + "\n")
                f.flush()
                print("Checkpoint saved")
    except KeyboardInterrupt:
        print("Training stopped. Going to save model")

    if train_to_backup:
        checkpoint_path = trainer.save()
        print(checkpoint_path)
        f = open(METHOD.get_checkpoint_path(method, reward), "a")
        f.write(checkpoint_path + "\n")
        f.close()
        save_model(trainer, method, reward)

    test_env.reset()
    ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    params_default = "params.list"
    parser.add_argument("-p", "--params", type=str, default=params_default)
    args = parser.parse_args()
    main(args)
