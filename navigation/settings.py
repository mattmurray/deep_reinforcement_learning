# settings
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # batch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # soft update of target params
LR = 5e-4  # nn learning rate
UPDATE_EVERY = 4  # frequency for updating network weights


# hyperparameters
N_EPISODES = 500
MAX_T = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.6
