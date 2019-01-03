# imports
import os
from datetime import datetime
from collections import deque
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from unityagents import UnityEnvironment

# reinforcement learning code
from settings import *
from noise import *
from agent import *
from replay import *
from model import *

# train on GPU when available
if torch.cuda.is_available():
    print("GPU available")
    device = torch.device("cuda:0")
    # device = "cuda:0"
else:
    print("No GPU available")
    device = torch.device("cpu")
    # device = "cpu"

# function for training agents
def train_agents(n_episodes, print_every, save_path=None):

    scores_deque = deque(maxlen=100)
    scores_all = []
    rolling_average = []

    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        states = np.reshape(states, (1, 48)) # 2 agents, both observing state of 24
        agent_0.reset()
        agent_1.reset()
        scores = np.zeros(num_agents)

        while True:
            action_0 = agent_0.act(states, ADD_NOISE)
            action_1 = agent_1.act(states, ADD_NOISE)
            actions = np.concatenate((action_0, action_1), axis=0)
            actions = np.reshape(actions, (1, 4))
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            next_states = np.reshape(next_states, (1, 48))
            rewards = env_info.rewards
            done = env_info.local_done

            agent_0.step(states, actions, rewards[0], next_states, done, 0)
            agent_1.step(states, actions, rewards[1], next_states, done, 1)
            scores += rewards
            states = next_states

            if np.any(done):
                break

        max_score = np.max(scores) # single score for episode
        scores_deque.append(max_score)
        average_score = np.mean(scores_deque) # average score over 100 episodes (+0.5 == solved)

        # data for chart / analysis later
        scores_all.append(max_score)
        rolling_average.append(average_score)

        current_eps = agent_0.eps
        print('\rEpisode {}\tAverage Score: {:.3f}\tMax Score: {:.3f}\tEPS: {:.3f}'.format(i_episode, average_score, max_score, current_eps), end="")

        if i_episode % print_every == 0 or (average_score > 1.0 and len(scores_deque) >= 100):
            print('\rEpisode {}\tAverage Score: {:.3f}\tMax Score: {:.3f}\tEPS: {:.3f}'.format(i_episode, average_score, max_score, current_eps))

            if save_path is not None:
                for agent_id, agent in enumerate([agent_0, agent_1]):
                    torch.save(agent.actor_local.state_dict(), os.path.join(save_path, 'checkpoint_actor_local_' + str(agent_id) + '.pth'))
                    torch.save(agent.critic_local.state_dict(), os.path.join(save_path, 'checkpoint_critic_local_' + str(agent_id) + '.pth'))

        if average_score > 0.5:
            break

    return scores_all, rolling_average


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

# # initialize agents
""" Setup two independent agents with shared experience memory """
agent_0 = Agent(state_size=state_size,
                action_size=action_size,
                num_agents=1,
                random_seed=0,
                device=device,
                lr_actor=LR_ACTOR,
                lr_critic=LR_CRITIC,
                weight_decay_critic=WEIGHT_DECAY_CRITIC,
                batch_size=BATCH_SIZE,
                buffer_size=BUFFER_SIZE,
                gamma=GAMMA,
                tau=TAU,
                update_every=UPDATE_EVERY,
                n_updates=N_UPDATES,
                eps_start=EPS_START,
                eps_end=EPS_END,
                eps_decay=EPS_DECAY)

agent_1 = Agent(state_size=state_size,
                action_size=action_size,
                num_agents=1,
                random_seed=0,
                device=device,
                lr_actor=LR_ACTOR,
                lr_critic=LR_CRITIC,
                weight_decay_critic=WEIGHT_DECAY_CRITIC,
                batch_size=BATCH_SIZE,
                buffer_size=BUFFER_SIZE,
                gamma=GAMMA,
                tau=TAU,
                update_every=UPDATE_EVERY,
                n_updates=N_UPDATES,
                eps_start=EPS_START,
                eps_end=EPS_END,
                eps_decay=EPS_DECAY)

# create directory for the run
runs_path = "runs"
timestamp = datetime.strftime(datetime.now(), "%Y%m%d%H%M")
run_path = os.path.join(runs_path, timestamp)
os.mkdir(run_path)

# train agents
scores, scores_avg = train_agents(n_episodes=2500,
                                  print_every=PRINT_EVERY,
                                  save_path=run_path)

# close the environment
env.close()

# save scores to CSV
score_data = pd.DataFrame(scores, columns=['episode_score'])
score_data['scores_100_avg'] = scores_avg
score_data['episode_number'] = score_data.index + 1
data_file_name = "{}.csv".format(timestamp)
print("\nSaving scores data to:\n{}".format(data_file_name))
score_data.to_csv(os.path.join(run_path, data_file_name), index=None)

# save chart of progress
chart_file_name = "{}.png".format(timestamp)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.plot(np.arange(1, len(scores)+1), scores_avg)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(os.path.join(run_path, chart_file_name))
print("Saved chart to:\n{}".format(chart_file_name))


