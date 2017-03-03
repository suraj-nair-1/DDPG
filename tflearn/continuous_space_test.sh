#!/bin/bash
./bin/HFO --offense-agents=1  --defense-npcs=1 --no-sync --fullstate &
sleep 5
python ../DDPG/tflearn/continuous_space_test_agent.py 6000 &
# sleep 5
# python ../DDPG/tflearn/continuous_space_test_agent.py 6000 &

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait
