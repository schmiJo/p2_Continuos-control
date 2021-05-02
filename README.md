# Continuous Project Jonas Schmidt

Welcome to the Continuous Control Project. <br>
In this Project we train an agent to follow a region with a robotic arm<br>
👇🏼This is the resulting agent<br>
![image info](./drawables/trained-agent.gif)

## Project Details

This project contains a solution to the second project of Udacity Deep Reinforcement Learning. This Project uses a DDPG
Algorithm to train the agent.

A reward of ~0.04 is provided for being within the ball, and a reward of 0 is provided for being outside the goal at any given timestep.
Therefore the goal is to follow the goal area with the robotic arm.


### State and Action Spaces

The state space consists of 33 dimensions and is continuous.

The action space consists of 4 continuous actions, which control the arm. 

The Agent is trained using a DDPG algorithm. For further information on training please read the Report.md.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

Check the following link for more details: <br>
<https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation>

## Getting Started

###Prerequisites
Python 3.6
Unity
Conda

##Installation:

1. Clone the repository
```
https://github.com/schmiJo/p2_Continuos-control
```
2. Install Jupyter Notebook
```
pip install jupyter
```
3. Create and activate a new environment for Python 3.6
* Linux or Mac
```
conda create --name drlnd python=3.6
source activate drlnd
```
* Windows
```
conda create --name drlnd python=3.6
activate drlnd
```
4. Install several dependencies 
```
pip install -r requirements.txt
```
5. Before running the Continuous_Control.ipynb change the kernel to match the drlnd environment by using the drop down Kernel menu.


Download the unity environment using the following link for macOs: <br>
https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip


Mre instructions for the installation can be found under: <br>
https://github.com/udacity/deep-reinforcement-learning#dependencies




