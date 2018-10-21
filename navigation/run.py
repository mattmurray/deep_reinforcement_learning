## imports

from collections import deque
import random
import torch
import argparse
import os
from unityagents import UnityEnvironment
import numpy as np
import pandas as pd
from datetime import datetime

# app imports
from agent import *
from model import *
from settings import N_EPISODES, MAX_T, EPS_START, EPS_END, EPS_DECAY
from settings import BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY

def arg_parser(args):

    try:
        n_episodes = int(args.n_episodes[0])
    except:
        n_episodes = N_EPISODES
    try:
        max_t = int(args.max_t[0])
    except:
        max_t = MAX_T
    try:
        eps_start = float(args.eps_start[0])
    except:
        eps_start = EPS_START
    try:
        eps_end = float(args.eps_end[0])
    except:
        eps_end = EPS_END
    try:
        eps_decay = float(args.eps_decay[0])
    except:
        eps_decay = EPS_DECAY

    print('\nPARAMS:\nn_episodes: {}\nmax_t: {}\neps_start: {}\neps_end: {}\neps_decay: {}\n'.format(
        n_episodes, max_t, eps_start, eps_end, eps_decay))

    return n_episodes, max_t, eps_start, eps_end, eps_decay


def dqn(env, n_episodes, max_t, eps_start, eps_end, eps_decay, save_path=None):
    """Deep Q-Learning.

    Params
    ======
    env (Unity Environment): environment instance
    n_episodes (int): maximum number of training episodes
    max_t (int): maximum number of timesteps per episode
    eps_start (float): starting value of epsilon, for epsilon-greedy action selection
    eps_end (float): minimum value of epsilon
    eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    solved = False
    solved_in = 0

    for i_episode in range(1, n_episodes + 1):
        # Reset everything to begin episode
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            # Choose action
            action = agent.act(state, eps)

            # Apply action and receive environment feedback
            env_info = env.step(int(action))[brain_name]

            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            # Update model
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        # Save episode result and decrease epsilon
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        # Log progress
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 50 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if np.mean(scores_window) >= 13.0 and not solved:
            solved = True
            solved_in = i_episode - 100

            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))

            if save_path == None:
                # save to current directory
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            else:
                # save to specified directory
                torch.save(agent.qnetwork_local.state_dict(), os.path.join(save_path, 'checkpoint.pth'))

    return scores, solved_in


# parse parameters from command line
parse = argparse.ArgumentParser()
parse.add_argument('-n', '--n_episodes', help='number of episodes in training', nargs=1)
parse.add_argument('-m', '--max_t', help='max number of timesteps per episode', nargs=1)
parse.add_argument('-s', '--eps_start', help='starting value of epsilon', nargs=1)
parse.add_argument('-e', '--eps_end', help='minimum value of epsilon', nargs=1)
parse.add_argument('-d', '--eps_decay', help='multiplicative factor (per episode) for decreasing epsilon', nargs=1)

# set parameters for agent training
n_episodes, max_t, eps_start, eps_end, eps_decay = arg_parser(parse.parse_args())

# start environment
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions available to the agent
action_size = brain.vector_action_space_size

# examine state space
state = env_info.vector_observations[0]

# get size of the state
state_size = len(state)

# Initialize the agent
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

# create directory for the run
runs_path = "runs"
timestamp = datetime.strftime(datetime.now(), "%Y%M%d%H%M")
run_path = os.path.join(runs_path, timestamp)
os.mkdir(run_path)

# train agent
scores, solved_in = dqn(env,
                        n_episodes=n_episodes,
                        max_t=max_t,
                        eps_start=eps_start,
                        eps_end=eps_end,
                        eps_decay=eps_decay,
                        save_path=run_path)

# close environment
env.close()

# save scores to CSV
score_data = pd.DataFrame(scores, columns=['episode_score'])
score_data['episode_number'] = score_data.index + 1

file_name = "{},solved_in__{},n_episodes__{},max_t__{},eps_start__{},eps_end__{},eps_decay__{},batch_size__{},buffer_size__{},gamma__{},tau__{},lr__{},update_every__{}.csv".format(timestamp, solved_in, n_episodes, max_t, eps_start, eps_end, eps_decay, BATCH_SIZE, BUFFER_SIZE, GAMMA, TAU, LR, UPDATE_EVERY)
print("Saving scores data to:\n{}".format(file_name))
score_data.to_csv(os.path.join(run_path, file_name), index=None)
