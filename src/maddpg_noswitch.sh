#!/bin/bash
./bin/HFO --no-sync  --frames-per-trial=500 --untouched-time=500 --offense-agents=2 --defense-agents=1 --agent-play-goalie --port 6000 --no-logging --seed 988 &
sleep 5
python ../DDPG/src/maddpg_noswitch.py 6000 1 1 989 &
sleep 5
python ../DDPG/src/maddpg_noswitch.py 6000 1 2 990 &
sleep 5
python ../DDPG/src/maddpg_noswitch.py 6000 0 3 990 &
# sleep 5
# python ../DDPG/tflearn/continuous_space_test_agent.py 6000 &

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

