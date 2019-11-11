[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


# Project 2: Continuous Control


### Overview

![Trained Agent][image1]

Source: Udacity

This project works with Unity's [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

In this environment, it involves getting 20 double-jointed robotic arms to move to target locations and make contact with the spheres in the environment

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The goal is for the agents to solve the environment and achive an average score of 30.0 over 100 consecutive episodes.

### Distributed Training

There are two versions of the Unity enviornment: 
* ***option 1:*** Involves a single agent.
* ***option 2:*** Involves 20 identical agents, each with its own copy of the environment.
This project solves option 2 to get the 20 agents (robotic arms) to make contact with the spheres in the environment. 

Option 2 is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

* ***Option 1: Solve the First Version for a single agent***

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

* ***Option 2: Solve the Second Version for 20 agents***

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, the rewards that each agent received are added up  (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores and then the average of the 20 scores are taken to yield an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Repository Contents
This repository contains the files listed below which were successfully used to train the robotic arms.
1. ***Continuous_Control.ipynb:*** This file contains the starter code from Udacity's Repository and the implementation of the DQN Function used to train the agent.
2. ***ddpg_agent.py:*** This file contains the DQN Agent class.
3. ***model.py:*** This file contains defined QNetwork architecture.
4. ***checkpoint_actor.pth:*** This file contains the trained and saved Actor Network weights.
5. ***checkpoint_critic.pth:*** This file contains the trained and saved Critic Network weights.
6. ***README.md:*** This file contains the overview of this project and how to understand its contents and implement one of your own.
7. ***report.md:*** This file contains a report of the learning algorithm used in this project.

### Instructions

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent! 

### Dependencies

The following are requirments to setup and run the code of this repository:
* ***python 3***
* ***numpy***
* ***PyTorch:*** Installation instructions [click here](https://pytorch.org/get-started/locally/)
* ***Unity ML-Agents:*** Installation instructions [click here](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Installing-Unity-ml-agents-on-Windows)
* ***NVIDIA drivers on local machine***


### Getting Started

To get started on your own system and train an agent of your own similar to what is done here, follow the instructions below:
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 


### Original Udacity Code

To try out your own implementation, the original Udacity repository for this project can be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).