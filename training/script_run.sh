#!/bin/bash

export PYTHONPATH=$PYTHONPATH:~/git/obs_randompat_MARL
source ~/git/obs_randompat_MARL/venv/bin/activate
python3 ~/git/obs_randompat_MARL/training/pbt.py -p ~/git/obs_randompat_MARL/training/params_pbt_cluster.list -n pbt_9