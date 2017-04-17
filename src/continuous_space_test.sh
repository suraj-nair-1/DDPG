#!/bin/bash
./bin/HFO --frames-per-trial=500 --untouched-time=500 --offense-agents=2 --defense-npcs=1 --headless --fullstate --port 6200 --no-logging --seed 6350 &
sleep 5
python ../DDPG/src/continuous_space_test_agent.py 6200 1 6350 &
sleep 5
python ../DDPG/src/continuous_space_test_agent.py 6200 2 7350 &
#python ~/git/DDPG/tflearn/continuous_space_test_agent.py 6000 &

# sleep 5
# python ../DDPG/tflearn/continuous_space_test_agent.py 6000 &

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait
