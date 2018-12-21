BUFFER_SIZE = 100000    # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001             # for soft update of target parameters
LR_ACTOR = 0.0002       # learning rate of the actor
LR_CRITIC = 0.001       # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

N_EPISODES = 500
MAX_T = 1000
PRINT_EVERY = 10

ENVIRONMENT_PATH = 'Reacher_Windows_x86_64/Reacher.exe'