#!/bin/bash
./bin/HFO --headless --frames-per-trial=500 --untouched-time=500 --record --fullstate --offense-agents=2 --defense-npcs=1 --seed $1 --port $2 &
sleep 5
python ./../DDPG/pytorch_codebase/train_agents.py $2 &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

