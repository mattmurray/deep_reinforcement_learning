# Navigation task

## The environment

For this project, the goal is to train an agent to navigate and collect bananas in a large, square world.

##### Rewards:
- +1 for collecting a yellow banana.
- -1 for collecting a blue banana.

The goal is for the agent to collect as many yellow bananas as possible while avoiding the blue bananas.

##### State space:

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects aroudn the agent's forward direction.

Given this information, the agent has to learn how to best select actions.

##### Actions:

Four discrete actions are available, corresponding to:

- 0 - move forward
- 1 - move backward
- 2 - turn left
- 3 - turn right

##### Solving the environment:

The task is episodic, and in order to solve the environment, the agent must achieve an average score of +13.00 over 100 consecutive periods.

## Downloading the environment

The environment is already built, and can be downloaded from one of the following links:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip "Linux environment")
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip "Mac environment")
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip "Win 32 environment")
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip "Win 64 environment")

Once downloaded, unzip the contents into this directory. You should now have a new directory containing the environment files.

For example, if using 64-bit Windows, you will see a new folder named *Banana_Windows_x86_64*.

Finally, open up *settings.py* and update *ENVIRONMENT_PATH* to match your set-up. By default, it is set-up for a Windows 64-bit installation.

## Dependencies

See the [dependencies](../README.md "README.md") in the main repo.

## Training agents

1. Default settings:

    To train an agent with the default settings as specified in *settings.py*, run:

    ```bash
    python run.py
    ```

2. Custom settings:

    You can alter some of the default settings without having to alter the *settings.py* file, by passing the following arguments:
    
    - -n: Changes the n_episodes parameter.
    - -m: Changes the max_t parameter
    - -s: Changes the eps_start parameter.
    - -e: Changes the eps_end parameter.
    - -d: Changes the eps_decay parameter.

    For example, the below will overwrite the default settings for eps_start and eps_decay to 0.9 and 0.7, respectively:

    ```bash
    python run.py -s 0.9 -d 0.7
    ```


When running the script, an agent will be trained, and the scores achieved in the episodes will be saved into a new subdirectory inside the *runs* directory, named as the current timestamp, for example *runs/201810221119*.
 
Provided that the agent is able to reach the target score within the number of episodes specified when training, the final pytorch weights will also be saved in the directory as *checkpoint.pth*.

