#!/bin/bash
./bin/HFO --frames-per-trial=500 --untouched-time=500 --offense-agents=2 --defense-npcs=1 --headless --fullstate --port 6600 --no-logging --seed 2312 &
sleep 5
python ../DDPG/src/switchagent.py 6600 1 6371 &
sleep 5
python ../DDPG/src/switchagent.py 6600 2 7371 &
#python ~/git/DDPG/tflearn/continuous_space_test_agent.py 6000 &

# sleep 5
# python ../DDPG/tflearn/continuous_space_test_agent.py 6000 &

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait
