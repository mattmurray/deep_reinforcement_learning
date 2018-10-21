# default settings and hyperparameters

BUFFER_SIZE = int(1e5)          # replay buffer size
BATCH_SIZE = 64                 # batch size
GAMMA = 0.99                    # discount factor
TAU = 1e-3                      # soft update of target params
LR = 5e-4                       # nn learning rate
UPDATE_EVERY = 4                # frequency for updating network weights

N_EPISODES = 500                # maximum number of training episodes to run
MAX_T = 1000                    # maximum number of timesteps per episode
EPS_START = 1.0                 # starting value of epsilon
EPS_END = 0.01                  # ending value of epsilon
EPS_DECAY = 0.995               # epsilon decay factor
