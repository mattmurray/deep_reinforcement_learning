# Continuous Control

### The environment
---

This environment in this project is Unity's Reacher environment [(see here)](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) where double-joined arms move to reach moving balloons.

- **Number of agents:** 20
- **State size:** 33
- **Action space:** Continuous, of size 4 (corresponding to torque applicable to two joints)
- **Agent rewards:** +0.1 for each step that an Agent's hand is in the goal location.
- **Solved when:** Agents achieve an average score of +30 (over 100 consecutive episodes, and over all agents).

---


#### Downloading the environment

The environment can be downloaded from one of the following links:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Once downloaded, unzip the contents to this directory and update *ENVIRONMENT_PATH* inside *settings.py* with the path to your environment.

---

#### Dependencies

See the [dependencies](../README.md "README.md") in the main repo.

---
## Training agents

1. Default settings:

    To train agents with the default settings as specified in *settings.py*, run:

    ```bash
    python run.py
    ```

2. Custom settings:

    You can alter some of the default settings without having to alter the *settings.py* file, by passing the following arguments:
    
    - **-n**: Changes the **n_episodes** parameter.
    - **-m**: Changes the **max_t** parameter
    - **-b**: Changes the **buffer_size** parameter.
    - **-g**: Changes the **gamma** parameter.
    - **-t**: Changes the **tau** parameter.
    - **-a**: Changes the **lr_actor** parameter.
    - **-c**: Changes the **lr_critic** parameter.
    - **-w**: Changes the **weight_decay** parameter.



When running the script, the scores achieved and model weights will be saved into a new subdirectory inside the *runs* directory, named as the current timestamp, for example *runs/201810221119*.
 
