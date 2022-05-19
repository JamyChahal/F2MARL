import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np


def main(args=None):
    max_simu_time = 30 * 60  # Sec
    nbr_exp_per_config = 30

    is_windows = False  # If linux, other command to do

    # Struct :
    #     map_length
    #     obs_range
    #     com_range
    #     nbr_robot
    #     discretization
    #     method
    #     strategy : CMAES;RR;CR;CC;A_CMOMMT;O_CMOMMT
    obs_range = 3
    com_range = 4
    map_size = 15
    discrete = 3
    #tot_agent = [1, 2, 3, 4, 5]
    #tot_target = [1, 2, 3, 4, 5, 6, 7]
    tot_agent = [2, 3, 5]
    tot_target = [3, 5, 7, 10]
    agent_speed = [2]
    target_speed = [1]

    config = []
    #strategies = ["I_CMOMMT", "A_CMOMMT", "RR", "CR", "CC"]
    methods = ["A_CMOMMT", "RANDOM", "NN_co", "I_CMOMMT"]
    #methods = ["I_CMOMMT"]
    for agent in tot_agent:
        for target in tot_target:
            if target >= agent:
                for method in methods:
                    for a_s in agent_speed:
                        for w_s in target_speed:
                            config.append([obs_range, com_range, agent, discrete, method, target, a_s, w_s, 1])
    debug = True
    debug_cmd = True

    print("Starting the simulation ..")

    for obs_range, com_range, nbr_agent, discretization, method, nbr_target, \
        agent_speed, target_speed, safety_range in config:

        print("Nbr of robot : " + str(nbr_agent))
        if is_windows:
            cmd = "python " + str(Path(__file__).resolve().parents[1]) + "\run_from_script.py"
        else:
            # TODO
            cmd = "python3 run_reactive_from_script.py"

        # Adding parameters
        cmd += " -t " + str(max_simu_time) + " -m " + str(map_size) + " -o " + str(obs_range) + " -c " + \
               str(com_range) + " -a " + str(nbr_agent) + " -d " + str(discretization) + " --method " + str(method)
        cmd += " -w " + str(nbr_target) + " --max_agent_speed " + str(agent_speed) + " --max_target_speed " + str(target_speed) + \
                " -e " + str(nbr_exp_per_config) + " -s " + str(safety_range)

        cmd += " --not_game_displayed --backup"

        if debug_cmd:
            cmd += " --debug"
        else:
            cmd += " --no_debug"

        if debug:
            print("CMD : " + cmd)

        if debug:
            if is_windows:
                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        else:
            if is_windows:
                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL
                )
            else:
                proc = subprocess.Popen(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        preexec_fn=os.setsid)
        try:
            while proc.poll() is None:
                continue
        except KeyboardInterrupt:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # Send the signal to all the process groups
            print("Wait 2sec to kill everything")
            time.sleep(2)

    print("Ending")


if __name__ == "__main__":
    main()
