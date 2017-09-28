#!/bin/bash
./bin/HFO --frames-per-trial=500 --untouched-time=500 --fullstate --no-sync --offense-agents=2 --defense-npcs=1 --no-logging --seed 4343 --port 4500 &
sleep 5
python ./../DDPG/train_agents.py 2 &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

