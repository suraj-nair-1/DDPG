#!/bin/bash
<<<<<<< HEAD
<<<<<<< HEAD
./bin/HFO --frames-per-trial=500 --untouched-time=500 --fullstate --headless --offense-agents=2 --defense-agents=1 --agent-play-goalie --no-logging --seed 5111 --port 5500 &
=======
./bin/HFO --frames-per-trial=500 --untouched-time=500 --fullstate --no-sync --offense-agents=2 --defense-agents=1 --agent-play-goalie --no-logging --seed 4343 --port 4500 &
>>>>>>> 4a539a1f8f97b2071f01b348c19db656633a0197
=======
./bin/HFO --frames-per-trial=500 --untouched-time=500 --fullstate --no-sync --offense-agents=2 --defense-agents=1 --agent-play-goalie --no-logging --seed 4343 --port 4500 &
>>>>>>> 4a539a1f8f97b2071f01b348c19db656633a0197
sleep 5
./../DDPG/src/maddpg_switch.py 5500 1 1 5111 &
sleep 5
./../DDPG/src/maddpg_switch.py 5500 1 2 5111 &
sleep 5
<<<<<<< HEAD
<<<<<<< HEAD
./../DDPG/src/maddpg_switch.py 5500 0 3 5111 &
leep 5
=======
./../DDPG/src/maddpg_switch.py 4500 0 3 4343 &
sleep 5
>>>>>>> 4a539a1f8f97b2071f01b348c19db656633a0197
=======
./../DDPG/src/maddpg_switch.py 4500 0 3 4343 &
sleep 5
>>>>>>> 4a539a1f8f97b2071f01b348c19db656633a0197
# python ../DDPG/tflearn/continuous_space_test_agent.py 6000 &

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

