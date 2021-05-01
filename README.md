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

##Installation:

Clone the repsoitory
```
https://github.com/schmiJo/p2_Continuos-control
```
Install Jupyter Notebook
```
pip install jupyter
```

Then open the notebook Continuous_Control.ipynb

Download the unity environment using the following link for macOs: <br>
https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip




