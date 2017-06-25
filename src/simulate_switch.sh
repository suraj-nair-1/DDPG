#!/bin/bash
./bin/HFO --frames-per-trial=500 --untouched-time=500 --offense-agents=2 --defense-npcs=1 --no-sync --fullstate --port 4200 --seed 605 &
sleep 5
python ../DDPG/src/simulate_switch.py 4200 1 targetcloser12_1_3000000.0.tflearn targetfarther12_1_3000000.0.tflearn &
sleep 5
python ../DDPG/src/simulate_switch.py 4200 2 targetcloser12_2_3000000.0.tflearn targetfarther12_2_3000000.0.tflearn &
#python ~/git/DDPG/tflearn/continuous_space_test_agent.py 6000 &

# sleep 5
# python ../DDPG/tflearn/continuous_space_test_agent.py 6000 &

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait
