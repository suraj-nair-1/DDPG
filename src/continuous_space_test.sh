#!/bin/bash
./bin/HFO --frames-per-trial=500 --untouched-time=500 --offense-agents=1 --no-sync --fullstate --port 4200 &
sleep 5
python ../DDPG/src/continuous_space_test_agent.py 4200 &
#python ~/git/DDPG/tflearn/continuous_space_test_agent.py 6000 &

# sleep 5
# python ../DDPG/tflearn/continuous_space_test_agent.py 6000 &

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait
