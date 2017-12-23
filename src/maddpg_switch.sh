#!/bin/bash
./bin/HFO --frames-per-trial=500 --untouched-time=500 --fullstate --headless --offense-agents=2 --defense-agents=1 --agent-play-goalie --no-logging --seed 5111 --port 5500 &
sleep 5
./../DDPG/src/maddpg_switch.py 5500 1 1 5111 &
sleep 5
./../DDPG/src/maddpg_switch.py 5500 1 2 5111 &
sleep 5
./../DDPG/src/maddpg_switch.py 5500 0 3 5111 &
leep 5
# python ../DDPG/tflearn/continuous_space_test_agent.py 6000 &

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

