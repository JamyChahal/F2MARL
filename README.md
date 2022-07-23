# F2MARL

## Notice

The code has been released in open source, however is still under development. Some part of the code will be cleaned up soon. 

## Publication

Original code from the paper "A force field multiagent reinforcement learning for tracking mobile targets" published in the conference IJCAI2022 https://ijcai-22.org/

Reference arriving soon. Please cite this paper if you use or get inspired from this code. 

## Installation

`pip3 install -r requirements.txt`

## Execution

### Simple run of the environment 

In the **evaluation** folder, execute *run.py* to have a simple view of the simulation : 

* The file *params.list* is used to modify all the parameters of the simulation
* To change the agent's method strategy, do the modification at line 143, with the possibility of :
  * METHOD.I_CMOMMT : I-CMOMMT method
  * METHOD.A_CMOMMT : A-CMOMMT method
  * METHOD.RANDOM : random behavior
  * METHOD.NN_CO : model trained, the location is precised at line 100

### Run a batch of simulation 

In the evaluation folder, execute *script.py* or *script_multithread.py* to run several simulations and extract the result in the file *result.txt*

## Training

### Single set of hyperparameters training 

In the **training** folder, execute *train.py*. The parameters for the training are specified in the file *params.list* . When stopping the run, the model is saved in the same folder. 

### PBT training

In the **training** folder, execute *pbt.py*. The parameters for the training are specified in the file *params_pbt.list* . The model will be saved in ~/ray_result/{name_of_training}/ once one instance of the pbt reached the condition specified at line 166.
