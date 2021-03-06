import argparse
import json
import os
import time


from behaviors.agent_A_CMOMMT import AgentACMOMMT
from behaviors.agent_A_CMOMMT_obs import AgentACMOMMTobs
from behaviors.agent_I_CMOMMT import AgentICMOMMT
from behaviors.agent_NN_randompat import AgentNNRP
from behaviors.agent_NN import AgentNN
from behaviors.outsideBehavior import OutsideBehavior
from behaviors.random_walking_policy import RandomBehavior
from behaviors.target_reactive import TargetReactive
from environment.mpe import obs_patrol_v0
from evaluation.Analyze import Analyze

from ray.rllib.utils.framework import try_import_tf
from enum import Enum, auto

from pathlib import Path

from evaluation.Analyze import Analyze

tf1, tf, tfv = try_import_tf()


class METHOD(Enum):
    OUTSIDE = auto()
    NONE = auto()
    RANDOM = auto()
    A_CMOMMT = auto()
    A_CMOMMT_OBS = auto()
    I_CMOMMT = auto()
    I_IRL_CMOMMT = auto()
    NN_co = auto()
    NN_ind = auto()

    @classmethod
    def is_trained(cls, method):
        return False if method == cls.OUTSIDE or method == cls.A_CMOMMT or method == cls.A_CMOMMT_OBS \
                        or method == cls.I_CMOMMT or method == cls.RANDOM else True

    @classmethod
    def is_implemented(cls, method):
        return True


def target_policy(observation, agent, targetEntities):
    # Agent observation: [TYPE X Y]
    # With [0 : self_pose, 1: distance with agent, 2: distance with adversary]
    # Agent action space: [speed btw [0, 1], angle btw [0, 2pi]]
    action = (0, 0)
    if "adversary" in agent:
        id = int(agent.split("adversary_", 1)[1])
        targetEntities[id].set_observation(observation)
        action = targetEntities[id].get_action()
    return action



def agent_policy(reward, observation, agent, agentEntities, method, centralized_learning):
    action = (0, 0)
    if "agent" in agent:
        id = int(agent.split("agent_", 1)[1])
        if not METHOD.is_trained(method):
            agentEntities[id].set_observation(observation)
            action = agentEntities[id].get_action()
        else:
            # action = agentEntities[id].do_prediction(reward, vectorization_obs(observation))
            #action = agentEntities[id].do_prediction(reward, vectorization_obs(observation, centralized_learning))
            action = agentEntities[id].do_action(reward, observation)
    return action


def read_last_line(filename):
    with open(filename, 'r') as f:
        last_line = f.readlines()[-1].rstrip('\n')
    return last_line


def get_agent_model(method):
    # https://github.com/ray-project/ray/blob/master/rllib/tests/test_export.py
    if METHOD.is_trained(method):
        script_dir = os.path.dirname(__file__)
        if method == METHOD.NN_co:
            #model_dir = "/home/jamy/git/obs_randompat_MARL/evaluation/models/model_pbt8"
            model_dir = os.path.join(script_dir + "/models/model_pbt8")
        if method == METHOD.NN_ind:
            model_dir = os.path.join(script_dir + "/models/model")
        model = tf.saved_model.load(model_dir)
        sign = model.signatures['serving_default']
        return sign
    else:
        return None



def backup_result(o_av_m=0, nbr_agent=0, nbr_target=0, obs_range=0, com_range=0, sim_duration=0,
                  map_size=0, method=METHOD.NONE, max_idleness=0, max_idleness_blur=0, A_metric=0,
                  agent_speed=0, target_speed=0, target_behavior="None", H=0):
    f = open("result.txt", "a+")
    f.write("\n")
    sep = ";"
    f.write(
        str(method) + sep + str(nbr_agent) + sep + str(nbr_target) + sep + str(obs_range) + sep + str(com_range) + sep
        + str(sim_duration) + sep + str(o_av_m) + sep + str(map_size) + sep + str(max_idleness) + sep +
        str(max_idleness_blur) + sep + str(A_metric) + sep + str(agent_speed) + sep + str(target_speed) + sep
        + str(target_behavior) + sep + str(H))
    f.close()


def main(args):
    # Parameters
    map_size = args.map_size
    obs_range = args.obs_range
    com_range = args.com_range
    nbr_agent = args.nbr_agent
    nbr_target = args.nbr_target
    nbr_episodes = args.nbr_episodes
    discretization = args.discretization
    max_cycles = args.sim_time
    gui = args.gui
    method_name = args.method
    is_backup = args.is_backup
    max_agent_speed = args.max_agent_speed
    max_target_speed = args.max_target_speed
    centralized_training = args.centralized_training
    target_behavior = args.target_behavior

    method = METHOD[method_name]

    if not METHOD.is_implemented(method):
        print("Method not yet implemented")
        print(method)
        exit()

    # Create tf1 session
    sess = tf1.Session()

    obs_to_normalize = METHOD.is_trained(method)
    has_protection_force = False
    env = obs_patrol_v0.env(nbr_agent=nbr_agent, nbr_target=nbr_target, obs_range=obs_range, com_range=com_range,
                            safety_range=safety_range,
                            map_size=map_size, obs_to_normalize=obs_to_normalize,
                            max_cycles=max_cycles, has_protection_force=has_protection_force,
                            is_reward_individual=False, gradual_reward=True, max_target_speed=max_target_speed,
                            max_agent_speed=max_agent_speed, is_reward_shared=True)

    analyze = Analyze(obs_range=obs_range, discretization=1, obs_to_normalize=obs_to_normalize, map_size=map_size,
                      nbr_target=nbr_target)
    # Create the target's behaviors
    targetEntities = []
    for i in range(0, nbr_target):
        if target_behavior == "evasive":
            targetEntities.append(TargetReactive(map_size=map_size, det_range=obs_range+2, is_in_training=False))
        elif target_behavior == "random":
            targetEntities.append(RandomBehavior(map_size=map_size))
        else:
            print("ERROR : target behavor is not evasive nor random. Exit simulation here.")
            exit()

    # Prepare the agent policy
    model = get_agent_model(method)

    agentEntities = []
    for i in range(0, nbr_agent):
        if method == METHOD.OUTSIDE:
            agentEntities.append(OutsideBehavior(map_size=map_size, id=i, max_speed=max_agent_speed))
        if method == METHOD.RANDOM:
            agentEntities.append(RandomBehavior(map_size=map_size))
        elif method == METHOD.A_CMOMMT:
            agentEntities.append(AgentACMOMMT(map_size=map_size, obs_range=obs_range, com_range=com_range))
        elif method == METHOD.I_CMOMMT:
            agentEntities.append(AgentICMOMMT(map_size=map_size, obs_range=obs_range, com_range=com_range, gamma=2*max_cycles))
        elif method == METHOD.NN_co or method == METHOD.NN_ind:
            agentEntities.append(AgentNNRP(sess=sess, model=model, map_size=map_size))
        elif method == METHOD.A_CMOMMT_OBS:
            agentEntities.append(AgentACMOMMTobs(map_size=map_size, obs_range=obs_range, com_range=com_range))

    env.reset()
    i = 0
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        analyze.add_pose(observation, agent)
        if done:
            env.reset()
            #print(analyze.get_observability_A())
            backup_result(method=method, nbr_agent=nbr_agent, nbr_target=nbr_target, obs_range=obs_range,
                          map_size=map_size, com_range=com_range, sim_duration=max_cycles, agent_speed=max_agent_speed,
                          target_speed=max_target_speed, A_metric=analyze.get_observability_A(),
                          o_av_m=analyze.get_o_av_m(), max_idleness=analyze.get_max_idleness(),
                          max_idleness_blur=analyze.get_max_idleness_blur(), target_behavior=target_behavior, H=analyze.get_diversity())
            i = i + 1
            analyze.reset()
            if i > nbr_episodes:
                exit()

        if "agent" in agent:
            action = agent_policy(reward, observation, agent, agentEntities, method, centralized_training)
            # if agent == "agent_0":
            #    print(reward)
        else:
            action = target_policy(observation, agent, targetEntities)
        env.step(action)

        if gui:
            env.render()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    map_size = 10  # 100
    obs_range = 5  # 2
    com_range = 3
    nbr_agent = 3
    nbr_target = 1
    nbr_episodes = 25
    discretization = 3
    simulation_time = 60 * 60  # 1000
    gui = False  # Default False
    method = "NN_co"  #
    is_backup = True
    debug = False  # Default False
    max_target_speed = 1
    max_agent_speed = 2
    safety_range = 1

    target_behavior = "random"
    centralized_training = False

    parser.add_argument("-t", "--sim_time", type=int, default=simulation_time)
    parser.add_argument("-m", "--map_size", type=int, default=map_size)
    parser.add_argument("-o", "--obs_range", type=int, default=obs_range)
    parser.add_argument("-c", "--com_range", type=int, default=com_range)
    parser.add_argument("-a", "--nbr_agent", type=int, default=nbr_agent)
    parser.add_argument("-w", "--nbr_target", type=int, default=nbr_target)
    parser.add_argument("-d", "--discretization", type=int, default=discretization)
    parser.add_argument("-s", "--safety_range", type=float, default=safety_range)
    parser.add_argument("-e", "--nbr_episodes", type=float, default=nbr_episodes)

    parser.add_argument("--method", type=str, default=method)

    parser.add_argument("--max_target_speed", type=float, default=max_target_speed)
    parser.add_argument("--max_agent_speed", type=float, default=max_agent_speed)

    parser.add_argument("--target_behavior", type=str, default=target_behavior)
    parser.add_argument("--is_centralized_training", dest="centralized_training", action="store_true")
    parser.set_defaults(centralized_training=centralized_training)

    parser.add_argument('--backup', dest='is_backup', action='store_true')
    parser.add_argument('--not_backup', dest='is_backup', action='store_false')
    parser.set_defaults(is_backup=is_backup)
    parser.add_argument('--game_displayed', dest='is_game_displayed', action='store_true')
    parser.add_argument('--not_game_displayed', dest='is_game_displayed', action='store_false')
    parser.set_defaults(gui=gui)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no_debug', dest='debug', action='store_false')
    parser.set_defaults(debug=debug)
    args = parser.parse_args()

    main(args)
