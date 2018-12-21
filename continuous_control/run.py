## imports
from collections import deque
import numpy as np
import argparse
from datetime import datetime
import pandas as pd
from unityagents import UnityEnvironment
import torch
import os

# app imports
from agents import *
from model import *
from settings import BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY
from settings import N_EPISODES, MAX_T, PRINT_EVERY, ENVIRONMENT_PATH # default values


## functions

# arg parser
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
        buffer_size = float(args.buffer_size[0])
    except:
        buffer_size = BUFFER_SIZE
    try:
        gamma = float(args.gamma[0])
    except:
        gamma = GAMMA
    try:
        tau = float(args.tau[0])
    except:
        tau = TAU
    try:
        lr_actor = float(args.lr_actor[0])
    except:
        lr_actor = LR_ACTOR
    try:
        lr_critic = float(args.lr_critic[0])
    except:
        lr_critic = LR_CRITIC
    try:
        weight_decay = float(args.weight_decay[0])
    except:
        weight_decay = WEIGHT_DECAY


    print('\nPARAMS:\nn_episodes: {}\nmax_t: {}\nbuffer_size: {}\ngamma: {}\ntau: {}\nlr_actor: {}\nlr_critic: {}\nweight_decay: {}'.format(
        n_episodes, max_t, buffer_size, gamma, tau, lr_actor, lr_critic, weight_decay))

    return n_episodes, max_t, buffer_size, gamma, tau, lr_actor, lr_critic, weight_decay







# training function
def ddpg(n_episodes, max_t, print_every, agents, save_path=None):
    scores_deque = deque(maxlen=print_every)
    scores = []

    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations  # get the current state (for each agent)
        agents.reset()

        score = np.zeros(num_agents)  # initialize the score (for each agent)
        for t in range(max_t):
            action = agents.act(state)
            env_info = env.step(action)[brain_name]  # send all actions to tne environment
            next_state = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            agents.step(state, action, rewards, next_state, dones)
            state = next_state  # roll over states to next time step
            score += rewards  # update the score (for each agent)

            if np.any(dones):
                break

        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        average_score = np.mean(scores_deque)

        # log progress
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score), end="")

        if i_episode % print_every == 0 or average_score > 30.0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, average_score))

            if save_path == None:
                # save to current directory
                torch.save(agents.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(agents.critic_local.state_dict(), 'checkpoint_critic.pth')
            else:
                # save to specified directory
                torch.save(agents.actor_local.state_dict(), os.path.join(save_path, 'checkpoint_actor.pth'))
                torch.save(agents.critic_local.state_dict(), os.path.join(save_path, 'checkpoint_critic.pth'))

            # end training after environment solved
            if average_score > 30.0:
                break

    return scores, i_episode


# parse parameters from command line
parse = argparse.ArgumentParser()
parse.add_argument('-n', '--n_episodes', help='number of episodes in training', nargs=1)
parse.add_argument('-m', '--max_t', help='max number of timesteps per episode', nargs=1)
parse.add_argument('-p', '--buffer_size', help='frequency of print statements of scores', nargs=1)
parse.add_argument('-g', '--gamma', help='gamma number', nargs=1)
parse.add_argument('-t', '--tau', help='tau number', nargs=1)
parse.add_argument('-a', '--lr_actor', help='actor learning rate', nargs=1)
parse.add_argument('-c', '--lr_critic', help='critic learning rate', nargs=1)
parse.add_argument('-w', '--weight_decay', help='weight decay', nargs=1)

# set parameters
n_episodes, max_t, buffer_size, gamma, tau, lr_actor, lr_critic, weight_decay = arg_parser(parse.parse_args())

# start environment
env = UnityEnvironment(file_name=ENVIRONMENT_PATH)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# initialize the agents
agents = Agents(state_size=state_size,
                action_size=action_size,
                num_agents=num_agents,
                random_seed=2,
                buffer_size=buffer_size,
                batch_size=BATCH_SIZE,
                gamma=gamma,
                tau=tau,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                weight_decay=weight_decay)

# create directory for the run
runs_path = "runs"
timestamp = datetime.strftime(datetime.now(), "%Y%m%d%H%M")
run_path = os.path.join(runs_path, timestamp)
os.mkdir(run_path)

# train
scores, solved_in = ddpg(n_episodes=n_episodes,
                         max_t=max_t,
                         print_every=PRINT_EVERY,
                         agents=agents,
                         save_path=run_path)

# close the environment
env.close()

# save scores to CSV
score_data = pd.DataFrame(scores, columns=['episode_score'])
score_data['episode_number'] = score_data.index + 1

file_name = "{},solved_in__{},n_episodes__{},max_t__{},batch_size__{},buffer_size__{},gamma__{},tau__{},lr_actor__{},lr_critic__{},weight_decay__{}.csv".format(timestamp, solved_in, n_episodes, max_t, BATCH_SIZE, buffer_size, gamma, tau, lr_actor, lr_critic, weight_decay)

print("Saving scores data to:\n{}".format(file_name))
score_data.to_csv(os.path.join(run_path, file_name), index=None)
