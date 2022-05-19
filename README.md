# LSTM_RL

main branch

* Succeed to make PPO work with two policies

DONE : 

* Set parameters directly to the env (obs_range, com_range)
* Get observation only to surrounding from the env
* Set rewards 
  * Negative if outside the world 

TODO : 

* Hyperparameter to setup
* Normalization of the input for the training process (add a general parameter into the env. to normalize if it is a training policy)

### How to handle branches: 

`git checkout <feature-branch>`
`git pull`
`git checkout <release-branch>`
`git pull`
`git merge --no-ff <feature-branch>`
`git push`
`git tag -a branch-<feature-branch> -m "Merge <feature-branch> into <release-branch>"`
`git push --tags`
`git branch -d <feature-branch>`
`git push origin :<feature-branch>`

**Memo linux**

 `python3 -m tensorboard.main --logdir=~/ray_results/python`

