#!/bin/bash
./bin/HFO --frames-per-trial=500 --untouched-time=500 --fullstate --no-sync --offense-agents=2 --defense-agents=1 --agent-play-goalie --no-logging --seed 4343 --port 4500 &
sleep 5
./../DDPG/src/maddpg_switch.py 4500 1 1 4343 &
sleep 5
./../DDPG/src/maddpg_switch.py 4500 1 2 4343 &
sleep 5
./../DDPG/src/maddpg_switch.py 4500 0 3 4343 &
sleep 5
# python ../DDPG/tflearn/continuous_space_test_agent.py 6000 &

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

