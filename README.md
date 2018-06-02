# Using Options with Multi-Agent Deep Deterministic Policy Gradients

## Requirements

python 3.6

pytorch >= 0.3.0

hfo 0.1.5

hdf5 1.10.1

h5py 2.7.1

If using GPU, CUDA 8 and CuDNN 7

## To Launch
From the HFO folder, run the train_agents.sh file to launch. Command looks like

'bash ../pytorch_codebase/train_agents.sh [random seed] [port] [lognum] [options]'

'bash ../pytorch_codebase/train_agents.sh 92 5000 9 1'

Models are saved to pytorch_models folder. To playback and existing model set the PLAYBACK flag to True and set the appropriate model to load in the train_agents.py file.

Logs are written to the logging folder to a file called logs[lognum].txt. It is actually a .h5 file which is read by the MADDPG_Metrics.ipynb.

## Visualization

The MADDPG_Metrics.ipynb notebook takes a path to a log file, and plots the rewards over time, losses, Q value for different actions, option selection, etc.
