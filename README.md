# A collection of Deep Reinforcement Learning projects

## Projects

This will be updated over time as I add to the repo with new projects.

#### [1. Navigation](navigation/)


## Dependencies

In order to run the projects in this repository, you will need to install the Unity Machine Learning Agents Toolkit (see [https://unity3d.ai](https://unity3d.ai "ML-Agents")) as well as the OpenAI gym.

#### Setting up a python environment

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Install ML-Agents toolkit:
    - Follow the instructions [here](https://unity3d.ai "ML-Agents").
    
4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```
    
5. Install the additional dependencies in the requirements.txt file:
    ```bash
    conda install --file requirements.txt
    ```
    
6. Before running code make sure to activate your environment.
    - In a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.
    - In the command line, first activate the environment as per point 1 above before running any scripts.
    
