import os
import signal
import subprocess
import time
from pathlib import Path

from subprocess import Popen

import numpy as np


def main(args=None):
    max_simu_time = 30 * 60  # Sec
    nbr_exp_per_config = 100

    is_windows = False  # If linux, other command to do

    # Struct :
    #     map_length
    #     obs_range
    #     com_range
    #     nbr_robot
    #     discretization
    #     method
    #     strategy : CMAES;RR;CR;CC;A_CMOMMT;O_CMOMMT
    obs_range = 4
    com_range = 4
    map_size = 30
    discrete = 3
    #tot_agent = [1, 2, 3, 4, 5]
    #tot_target = [1, 2, 3, 4, 5, 6, 7]
    tot_agent = [2, 3, 5]
    tot_target = [3, 5, 7]
    agent_speed = [2]
    target_speed = [1]
    target_behavior = ["random", "evasive"] #or "random", "evasive"

    config = []
    #strategies = ["I_CMOMMT", "A_CMOMMT", "RR", "CR", "CC"]
    methods = ["A_CMOMMT_OBS", "RANDOM", "NN_ind", "I_CMOMMT"]
    methods = ["I_CMOMMT", "A_CMOMMT", "RANDOM", "NN_co"]
    for t_b in target_behavior:
        for agent in tot_agent:
            for target in tot_target:
                if target >= agent:
                    for method in methods:
                        for a_s in agent_speed:
                            for w_s in target_speed:
                                config.append([obs_range, com_range, agent, discrete, method, target, a_s, w_s, 1, t_b])
    debug = True
    debug_cmd = True

    print("Starting the simulation ..")

    cmd_list = []

    for obs_range, com_range, nbr_agent, discretization, method, nbr_target, \
        agent_speed, target_speed, safety_range, target_behavior in config:

        print("Nbr of robot : " + str(nbr_agent))
        if is_windows:
            cmd = "python " + str(Path(__file__).resolve().parents[1]) + "\run_from_script.py"
        else:
            # TODO
            cmd = "python3 run_from_script.py"

        # Adding parameters
        cmd += " -t " + str(max_simu_time) + " -m " + str(map_size) + " -o " + str(obs_range) + " -c " + \
               str(com_range) + " -a " + str(nbr_agent) + " -d " + str(discretization) + " --method " + str(method)
        cmd += " -w " + str(nbr_target) + " --max_agent_speed " + str(agent_speed) + " --max_target_speed " + str(target_speed) + \
                " -e " + str(nbr_exp_per_config) + " -s " + str(safety_range) + " --target_behavior " + str(target_behavior)

        cmd += " --not_game_displayed --backup"

        cmd += " --is_centralized_training" #TODO : according to the method itself

        if debug_cmd:
            cmd += " --debug"
        else:
            cmd += " --no_debug"

        if debug:
            print("CMD : " + cmd)

        cmd_list.append(cmd)

    #Let's run everything once and for all
    try:
        while len(cmd_list) > 0:
            cmd_package = cmd_list[:]
            del cmd_list[:]
            procs = [Popen(i, shell=True) for i in cmd_package]
            for p in procs:
                p.wait()
    except KeyboardInterrupt:
        print("Wait 2sec to kill everything")
        time.sleep(2)

    print("Ending")


if __name__ == "__main__":
    main()
