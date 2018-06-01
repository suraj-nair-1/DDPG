import sys
import itertools
from hfo import *
from collections import namedtuple
import numpy as np
import threading
import torch
from torch.autograd import Variable
import multiprocessing
from memory import ReplayMemory, Experience, ExperienceOptions
import multiprocessing
from MADDPG import MADDPG, OMADDPG
import numpy as np
import torch as th
import time
import h5py
import copy
import traceback
import subprocess
from pympler import asizeof
import time
import gc


#####################################################################
# SET LOGPATH DEPENDING ON USER
#####################################################################
LOGPATH = "/cs/ml/ddpgHFO/DDPG/"  # CMSCLUSTER LOGPATH
# LOGPATH = "/Users/surajnair/Documents/Tech/research/MADDPG_HFO/"
# LOGPATH = "/Users/anshulramachandran/Documents/Research/yisong/"
# LOGPATH = "/home/anshul/Desktop/"
#####################################################################

LOGNUM = int(sys.argv[2])  # Log Number For This Experiment
PLAYBACK = False  # T/F Evauate Trained Policy

# TRAINING PARAMS
OPTIONS = int(sys.argv[3])  # True/False Use Options
N_OPTIONS = 2
PRIORITIZED = True  # Prioritized Replay T/F
MAX_EPISODES = 50000  # Max Episodes
MAX_EP_STEPS = 500  # Max episode length
ACTOR_LEARNING_RATE = .001  # Base learning rate for the Actor network
CRITIC_LEARNING_RATE = .001  # Base learning rate for the Critic Network
GAMMA = 0.99  # Discount factor
TAU = 0.001  # Soft target update param

# EPSILON GREEDY EXPLORATION
EPS_ITERATIONS_ANNEAL = 300000  # Anneal Over N Timesteps
EPS_GREEDY_INIT = 1.0
EPS_GREEDY_MIN = 0.1

# REPLAY BUFFER
capacity = 1000000  # Capacity of Replay Buffer
batch_size = 1024
eps_before_train = 10
GPUENABLED = False  # Use GPU or only CPU
PORT = int(sys.argv[1])  # Port on which HFO Server Runs
SEED = int(sys.argv[4])  # Random Seed

FloatTensor = torch.cuda.FloatTensor if False else torch.FloatTensor
# Command to restart server
server_launch_command = "./bin/HFO --headless --frames-per-trial=500 \
                        --untouched-time=500 --no-logging --fullstate \
                        --offense-agents=2 --defense-npcs=1 --seed " + \
                        sys.argv[4] + " --port " + str(PORT)


def reset_server():
    '''
    In the event that the server fails, cleanup existing server 
    and relaunch
    '''
    subprocess.call('killall -9 rcssserver', shell=True)
    time.sleep(60)
    subprocess.Popen(server_launch_command, shell=True)
    time.sleep(30)


def connect():
    '''
    Connect to the half field offense server
    '''
    hfo = HFOEnvironment()
    hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
                        'bin/teams/base/config/formations-dt', PORT,
                        'localhost', 'base_left', False)
    return hfo


def take_action_and_step(a, o, env, eps):
    '''
    Take an action and step the environment
    '''
    # If explore, choose random action and random parameters
    if (np.random.random_sample() <= eps) and (not PLAYBACK):
        acts = np.random.uniform(1, 10, 4)
        a[:4] = acts / np.sum(acts)
        a[4] = np.random.uniform(0, 100)
        a[5] = np.random.uniform(-180, 180)
        a[6] = np.random.uniform(-180, 180)
        a[7] = np.random.uniform(-180, 180)
        a[8] = np.random.uniform(0, 100)
        a[9] = np.random.uniform(-180, 180)

        if OPTIONS:
            o = np.random.randint(0, 2)

    # Convert Action to the format which server accepts
    index = np.argmax(a[:4])
    if index == 0:
        action = (DASH, a[4], a[5])
    elif index == 1:
        action = (TURN, a[6])
    elif index == 2:
        action = (TACKLE, a[7])
    else:
        action = (KICK, a[8], a[9])

    # Take action and step
    env.act(*action)
    terminal = env.step()  # terminal state of episode or not
    s1 = env.getState()

    if OPTIONS:
        return s1, terminal, th.FloatTensor(a), o
    else:
        return s1, terminal, th.FloatTensor(a)


def get_curr_state_vars(s1):
    '''
    Compute distance to ball, ball distance to goal, 
    and kickable status of ball given the high 
    level feature set.
    '''
    curr_ball_prox = s1[53]
    curr_kickable = s1[12]
    goal_proximity = s1[15]
    ball_dist = 1.0 - curr_ball_prox
    goal_dist = 1.0 - goal_proximity
    ball_ang_sin_rad = s1[51]
    ball_ang_cos_rad = s1[52]
    ball_ang_rad = np.arccos(ball_ang_cos_rad)
    if ball_ang_sin_rad < 0:
        ball_ang_rad *= -1.
    goal_ang_sin_rad = s1[13]
    goal_ang_cos_rad = s1[14]
    goal_ang_rad = np.arccos(goal_ang_cos_rad)
    if goal_ang_sin_rad < 0:
        goal_ang_rad *= -1.
    alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)
    # Law of Cosines
    curr_goal_dist = np.sqrt(ball_dist * ball_dist + goal_dist *
                             goal_dist - 2. * ball_dist * goal_dist * np.cos(alpha))

    return curr_ball_prox, curr_goal_dist, curr_kickable


def get_rewards(terminal, curr_ball_prox, curr_goal_dist, curr_kickable,
                old_ball_prox, old_goal_dist, old_kickable):
    '''
    Compute reward given computed featurs
    '''
    r = 0.0

    # If game has finished, calculate reward based on whether or not a goal
    # was scored
    goal_made = False
    if terminal != IN_GAME:
        if int(terminal) == 1:
            goal_made = True
            r += 5
    else:
        # Movement to ball
        r += (curr_ball_prox - old_ball_prox)
        # Ball Movement towards goal
        r += -3.0 * float(curr_goal_dist - old_goal_dist)

    return [r], goal_made


def run_process(maddpg, player_num, player_queue, root_queue, feedback_queue, startep):
    '''
    This process is called in parallel for each agent. Includes agent model interacting with
    the environment taking actions and logging states/actions/rewards. When model is updated 
    the updated model is sent to this process.

    The Half Field Offense simulator requires that each agent joins on a seperate port, and only
    works if the agents are running ins seperate processes. That is why we use this semi-complicated
    process structure for running agents.
    '''
    # Connect to environment
    env = connect()

    # Use different random seed for the two agents
    if player_num == 0:
        np.random.seed(SEED)
    else:
        np.random.seed(SEED * 2)

    # If server fails, rough estimate of the number of iterations to make sure epsilon greedy
    # stays at the right place. Hacky but does not need to be precise.
    ITERATIONS = startep * 500.0

    # Store num_goals for logging purposes.
    NUM_GOALS = 0.0

    for ep in range(startep, MAX_EPISODES):
        status = IN_GAME
        states1 = env.getState()

        # Episode Stats
        ep_reward = 0.0
        ep_ave_max_q = 0.0
        old_reward = 0
        critic_loss = 0.0
        ep_good_q = 0.0
        ep_bad_q = 0.0
        ep_move_q = 0.0
        ep_turn_q = 0.0
        ep_tackle_q = 0.0
        ep_kick_q = 0.0
        ep_updates = 0.0

        rr = np.zeros((1,))
        old_ball_proxs = np.zeros((1,))
        old_goal_dists = np.zeros((1,))
        old_kickables = np.zeros((1,))
        for j in range(MAX_EP_STEPS):
            # Grab the state features from the environment
            states = states1
            try:
                states = torch.from_numpy(states).float()
            except:
                states = states.float()
            states = Variable(states).type(FloatTensor)

            # Predict Action
            if OPTIONS:
                actions, o = maddpg.select_action(
                    states.unsqueeze(0), player_num)
            else:
                actions = maddpg.select_action(states.unsqueeze(0), player_num)
            actions = actions[0].data

            # Take action and step
            if OPTIONS:
                states1, terminal, actions, o = take_action_and_step(
                    actions.numpy(), o, env, max(0.1, 1 - ITERATIONS / EPS_ITERATIONS_ANNEAL))
            else:
                states1, terminal, actions = take_action_and_step(
                    actions.numpy(), -1, env, max(0.1, 1 - ITERATIONS / EPS_ITERATIONS_ANNEAL))

            # Get computed features
            curr_ball_proxs, curr_goal_dists, curr_kickables = get_curr_state_vars(
                states1)

            # Compute reward (except for first timestep of episode)
            action_rewards = np.zeros((1,))
            if j != 0:
                action_rewards, goal_made = get_rewards(terminal, curr_ball_proxs, curr_goal_dists,
                                                        curr_kickables, old_ball_proxs, old_goal_dists, old_kickables)
                if np.any(goal_made):
                    NUM_GOALS += 1

            # Save old computed features and state
            rr += action_rewards
            old_ball_proxs = curr_ball_proxs
            old_goal_dists = curr_goal_dists
            old_kickables = curr_kickables
            states1 = np.stack(states1)
            states1 = torch.from_numpy(states1).float()

            # Log the transition by sending to the the main process via Queue
            if OPTIONS:
                player_queue.put((states.data, actions, states1,
                                  action_rewards, terminal, rr, (ep, j), o))
            else:
                player_queue.put((states.data, actions, states1,
                                  action_rewards, terminal, rr, (ep, j)))
            states = states1

            ITERATIONS += 1

            # EPISODE IS OVER
            ###########################################################
            if terminal:
                print(terminal)
                print('| Reward: ', rr, " | Episode", ep)
                break

            # Get latest model if available
            try:
                maddpg = root_queue.get(block=False)
            except:
                pass

            # Timing Queue to prevent processes from racing ahead
            # of main process
            try:
                new = feedback_queue.get(timeout=0.1)
            except:
                pass

        # If server crashes clean up queues
        if terminal == 5:
            player_queue.close()
            root_queue.close()
            feedback_queue.close()
            gc.collect()
            print("SLEEPING UNTIL KILLED")
            time.sleep(3600)
            break


def extra_stats(maddpg, player_num, opt=0):
    '''
    Records extra metrics on the performance of the model
    '''

    # Samples a batch
    transitions = maddpg.memory.sample(batch_size)
    if OPTIONS:
        batch = ExperienceOptions(*zip(*transitions))
    else:
        batch = Experience(*zip(*transitions))
    s_batch = Variable(th.stack(batch.states))
    action_batch = Variable(th.stack(batch.actions))

    whole_state = s_batch.view(batch_size, -1)
    move_batch = action_batch.clone()
    turn_batch = action_batch.clone()
    tackle_batch = action_batch.clone()
    kick_batch = action_batch.clone()
    good_batch = action_batch.clone()
    bad_batch = action_batch.clone()

    # For each element in the batch create an example for the different
    # types of actions
    for ind, elem in enumerate(s_batch):
        move_batch[ind, player_num] = Variable(torch.FloatTensor(
            np.array([1, 0, 0, 0, np.random.uniform(0, 100), np.random.uniform(-180, 180), 0, 0, 0, 0])))

        turn_batch[ind, player_num] = Variable(torch.FloatTensor(
            np.array([0, 1, 0, 0, 0, 0, np.random.uniform(-180, 180), 0, 0, 0])))

        tackle_batch[ind, player_num] = Variable(torch.FloatTensor(
            np.array([0, 0, 1, 0, 0, 0, 0, np.random.uniform(-180, 180), 0, 0])))

        kick_batch[ind, player_num] = Variable(torch.FloatTensor(
            np.array([0, 0, 0, 1, 0, 0, 0, 0, np.random.uniform(0, 100), np.random.uniform(-180, 180)])))

        ball_angle_sin = elem[player_num][51]
        ang = np.degrees(np.arcsin(ball_angle_sin.data[0]))
        if ang > 0:
            bad_ang = ang - 180
        else:
            bad_ang = ang + 180

        good_batch[ind, player_num] = Variable(torch.FloatTensor(
            np.array([1, 0, 0, 0, 10, ang, 0, 0, 0, 0])))
        bad_batch[ind, player_num] = Variable(torch.FloatTensor(
            np.array([1, 0, 0, 0, 10, bad_ang, 0, 0, 0, 0])))

    move_batch = move_batch.view(batch_size, -1)
    turn_batch = turn_batch.view(batch_size, -1)
    tackle_batch = tackle_batch.view(batch_size, -1)
    kick_batch = kick_batch.view(batch_size, -1)
    good_batch = good_batch.view(batch_size, -1)
    bad_batch = bad_batch.view(batch_size, -1)

    # Get Q values for the different types of actions
    target_move = maddpg.critic_predict(
        whole_state, move_batch, player_num)
    target_turn = maddpg.critic_predict(
        whole_state, turn_batch, player_num)
    target_tackle = maddpg.critic_predict(
        whole_state, tackle_batch, player_num)
    target_kick = maddpg.critic_predict(
        whole_state, kick_batch, player_num)
    target_good = maddpg.critic_predict(
        whole_state, good_batch, player_num)
    target_bad = maddpg.critic_predict(whole_state, bad_batch, player_num)

    ep_move_q = target_move.mean().data.cpu().numpy()[0]
    ep_turn_q = target_turn.mean().data.cpu().numpy()[0]
    ep_tackle_q = target_tackle.mean().data.cpu().numpy()[0]
    ep_kick_q = target_kick.mean().data.cpu().numpy()[0]
    ep_good_q = target_good.mean().data.cpu().numpy()[0]
    ep_bad_q = target_bad.mean().data.cpu().numpy()[0]

    player_stats = [ep_move_q, ep_turn_q,
                    ep_tackle_q, ep_kick_q, ep_good_q, ep_bad_q]
    return player_stats


def run():
    '''
    The main process which manages the child agent processes and does model training,
    model saving, and logging.
    '''
    # Set initial parameters for num agents and feature dimensions
    # for state and action
    sp = multiprocessing
    n_agents = 2
    n_states = 77
    n_actions = 10
    start_ep = 0
    lg = LOGNUM

    # Creates log file and initializes it
    f = h5py.File(LOGPATH + 'logging/logs' + str(lg) +
                  '.txt', "w", libver='latest')
    stats_grp = f.create_group("statistics")
    if OPTIONS:
        dset_scaling_factor = 1
    else:
        dset_scaling_factor = 2
    dset_move = stats_grp.create_dataset(
        "ep_move_q", (n_agents, MAX_EPISODES), dtype='f')
    dset_turn = stats_grp.create_dataset(
        "ep_turn_q", (n_agents, MAX_EPISODES), dtype='f')
    dset_tackle = stats_grp.create_dataset(
        "ep_tackle_q", (n_agents, MAX_EPISODES), dtype='f')
    dset_kick = stats_grp.create_dataset(
        "ep_kick_q", (n_agents, MAX_EPISODES), dtype='f')
    dset_good = stats_grp.create_dataset(
        "ep_good_q", (n_agents, MAX_EPISODES), dtype='f')
    dset_bad = stats_grp.create_dataset(
        "ep_bad_q", (n_agents, MAX_EPISODES), dtype='f')
    dset_rewards = stats_grp.create_dataset(
        "ep_reward", (n_agents, MAX_EPISODES), dtype='f')
    dset_closs = stats_grp.create_dataset(
        "ep_closs", (dset_scaling_factor, MAX_EPISODES), dtype='f')
    dset_aloss = stats_grp.create_dataset(
        "ep_aloss", (dset_scaling_factor, MAX_EPISODES), dtype='f')
    if OPTIONS:
        dset_options = stats_grp.create_dataset(
            "ep_options", (n_agents * N_OPTIONS, MAX_EPISODES), dtype='f')
    dset_numdone = stats_grp.create_dataset("ep_numdone", data=np.array([-1]))
    f.swmr_mode = True  # NECESSARY FOR SIMULTANEOUS READ/WRITE

    # Creates queue
    q1 = sp.Queue()
    q2 = sp.Queue()
    r1 = sp.Queue()
    r2 = sp.Queue()
    fdbk1 = sp.Queue()
    fdbk2 = sp.Queue()

    # Create Model
    if OPTIONS:
        maddpg = OMADDPG(n_agents, n_states, n_actions,
                         batch_size, capacity, eps_before_train, N_OPTIONS)
    else:
        maddpg = MADDPG(n_agents, n_states, n_actions,
                        batch_size, capacity, eps_before_train)

    # If playing back
    if PLAYBACK:
        maddpg.load(LOGPATH, 20, 4500)

    # Create child processes and start
    p1 = sp.Process(
        target=run_process, args=(maddpg, 0, q1, r1, fdbk1, start_ep))
    p2 = sp.Process(
        target=run_process, args=(maddpg, 1, q2, r2, fdbk2, start_ep))
    p1.start()
    time.sleep(5)
    p2.start()
    print("Started")

    itr = 1
    if GPUENABLED:
        maddpg.to_gpu()

    try:
        # Logging Option Selection
        if OPTIONS:
            p1optcounts = {}
            p2optcounts = {}
            for opt in range(N_OPTIONS):
                p1optcounts[opt] = 0
                p2optcounts[opt] = 0

        while True:
            # Get transitions from player queue
            # State_t, Action, State_t+1, transition reward, terminal, episode
            # reward, episode, option #
            if OPTIONS:
                p1_sts, p1_acts, p1_sts1, p1_rws, terminal1, episode_rew1, ep1, o1 = q1.get()
                p2_sts, p2_acts, p2_sts1, p2_rws, terminal2, episode_rew2, ep2, o2 = q2.get()
            else:
                p1_sts, p1_acts, p1_sts1, p1_rws, terminal1, episode_rew1, ep1 = q1.get()
                p2_sts, p2_acts, p2_sts1, p2_rws, terminal2, episode_rew2, ep2 = q2.get()
            ep1, step1 = ep1
            ep2, step2 = ep2

            if OPTIONS:
                p1optcounts[o1] += 1
                p2optcounts[o2] += 1

            if not ((ep1 == ep2) and (step1 == step2)):
                p1.join()
                p2.join()
                p1.terminate()
                p2.terminate()
                del p1
                del p2

            print("MAIN LOOP", maddpg.episode_done)
            maddpg.episode_done = ep1
            start_ep = ep1
            # Save Model
            if (maddpg.episode_done > 0) and (maddpg.episode_done % 500 == 0) and (step1 == 0):
                if not PLAYBACK:
                    maddpg.save(LOGPATH, LOGNUM)

            # Clean up episodes states/actions/rewards into tensors
            # and save them in replay buffer
            sts = torch.stack([p1_sts, p2_sts])
            acts = torch.stack([p1_acts, p2_acts])
            sts1 = torch.stack([p1_sts1, p2_sts1])
            rws = np.stack([p1_rws, p2_rws])
            rws = torch.FloatTensor(rws)
            if OPTIONS:
                os = th.zeros(n_agents, N_OPTIONS).type(
                    FloatTensor)
                os[0, int(o1)] = 1
                os[1, int(o2)] = 1
                if GPUENABLED:
                    maddpg.memory.push(sts.cuda(), acts.cuda(),
                                       sts1.cuda(), rws.cuda(), os.cuda())
                else:
                    maddpg.memory.push(sts, acts, sts1, rws, os)
            else:
                if GPUENABLED:
                    maddpg.memory.push(sts.cuda(), acts.cuda(),
                                       sts1.cuda(), rws.cuda())
                else:
                    maddpg.memory.push(sts, acts, sts1, rws)

            ###################################################################
            # If server crashes empty queue and reboot server and child
            # processes
            ###################################################################
            if (terminal1 == 5) or (terminal2 == 5):
                try:
                    while True:
                        if OPTIONS:
                            p1_sts, p1_acts, p1_sts1, p1_rws, terminal1, episode_rew1, ep1, o1 = q1.get(
                                block=False, timeout=0.001)
                        else:
                            p1_sts, p1_acts, p1_sts1, p1_rws, terminal1, episode_rew1, ep1 = q1.get(
                                block=False, timeout=0.001)
                except:
                    pass
                try:
                    while True:
                        if OPTIONS:
                            p2_sts, p2_acts, p2_sts1, p2_rws, terminal2, episode_rew2, ep2, o2 = q2.get(
                                block=False, timeout=0.001)
                        else:
                            p2_sts, p2_acts, p2_sts1, p2_rws, terminal2, episode_rew2, ep2 = q2.get(
                                block=False, timeout=0.001)
                except:
                    pass
                p1.join()
                p2.join()
                p1.terminate()
                p2.terminate()
                del p1
                del p2
                print("PROCESSES TERMINATED")
                time.sleep(300)
                print("SERVER FAIL")
                reset_server()
                print("RESET SERVER")

                copy_maddpg = copy.deepcopy(maddpg)
                copy_maddpg.to_cpu()
                copy_maddpg.memory = None

                p1 = sp.Process(
                    target=run_process, args=(copy_maddpg, 0, q1, r1, fdbk1, start_ep))
                p2 = sp.Process(
                    target=run_process, args=(copy_maddpg, 1, q2, r2, fdbk2, start_ep))
                print("Started")
                p1.start()
                time.sleep(30)
                p2.start()
                continue
            ###################################################################

            ###################################################################
            # At The End of each episode log stats and update policy
            ###################################################################
            if (terminal1 != 0):
                # Write logs to the log file
                if len(maddpg.memory.memory) > batch_size:
                    p1_logstats = extra_stats(maddpg, 0)
                    p2_logstats = extra_stats(maddpg, 1)
                    all_logstats = np.stack([p1_logstats, p2_logstats])

                    dset_move[:, maddpg.episode_done] = all_logstats[:, 0]
                    dset_turn[:, maddpg.episode_done] = all_logstats[:, 1]
                    dset_tackle[:, maddpg.episode_done] = all_logstats[:, 2]
                    dset_kick[:, maddpg.episode_done] = all_logstats[:, 3]
                    dset_good[:, maddpg.episode_done] = all_logstats[:, 4]
                    dset_bad[:, maddpg.episode_done] = all_logstats[:, 5]
                    if OPTIONS:
                        for j1 in range(N_OPTIONS):
                            dset_options[
                                j1, maddpg.episode_done] = p1optcounts[j1]
                            dset_options[N_OPTIONS + j1,
                                         maddpg.episode_done] = p2optcounts[j1]
                        for opt in range(N_OPTIONS):
                            p1optcounts[opt] = 0
                            p2optcounts[opt] = 0
                    dset_move.flush()
                    dset_turn.flush()
                    dset_tackle.flush()
                    dset_kick.flush()
                    dset_good.flush()
                    dset_bad.flush()
                    if OPTIONS:
                        dset_options.flush()
                dset_rewards[:, maddpg.episode_done] = np.array(
                    [episode_rew1, episode_rew2]).reshape((1, 2))
                dset_rewards.flush()

                # If not playing back update policy
                if PLAYBACK:
                    c_loss, a_loss = None, None
                else:
                    c_loss, a_loss = maddpg.update_policy(prioritized=True)

                # Write the losses to the log file
                if c_loss is not None:
                    if OPTIONS:
                        c_loss = c_loss.data.cpu().numpy()
                        a_loss = a_loss.data.cpu().numpy()
                        dset_closs[0, maddpg.episode_done] = np.array(c_loss)
                        dset_aloss[0, maddpg.episode_done] = np.array(a_loss)
                    else:
                        for nm in range(n_agents):
                            dset_closs[
                                nm, maddpg.episode_done] = c_loss[nm].data.cpu().numpy()
                            dset_aloss[
                                nm, maddpg.episode_done] = a_loss[nm].data.cpu().numpy()
                    dset_closs.flush()
                    dset_aloss.flush()
                dset_numdone[0] = maddpg.episode_done
                dset_numdone.flush()

                # Creates a lightweight version of the model to send to the
                # processes
                copy_maddpg = copy.deepcopy(maddpg)
                copy_maddpg.to_cpu()
                copy_maddpg.memory = None
                r1.put(copy_maddpg)
                r2.put(copy_maddpg)
            ###################################################################

            # Timing queue for the child processes
            fdbk1.put(0)
            fdbk2.put(0)

            # training step every 10 steps
            if not PLAYBACK:
                if itr % 10 == 0:
                    c_loss, a_loss = maddpg.update_policy(prioritized=True)
            maddpg.steps_done += 1
            itr += 1
    except Exception as e:
        r1.put(None)
        r2.put(None)
        traceback.print_exc()
        raise e


if __name__ == '__main__':
    run()
