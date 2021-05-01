# Project 2 Continuous Control - Report

## Motivation and Results

The Resulting Agent behaves in the environment like this:<br>
![image info](./drawables/trained-agent.gif)

The agent follows the dot pretty good, however it does not try to follow the middle of the dot. In order to achieve an
agent who tries to stay in the middle of the target, the reward would have to be changed in such a way that the distance
to the middle of the dot is inversely proportional.

The Agent was trained within 220 episodes: <br>
![image info](./drawables/training.png)

## Learning Algorithm

Deep Deterministic Policy Gradient (DDPG) was used to train the agent.

### State and Action Spaces

The action space consists of 4 different continuous actions. These four actions control the robotic arm.

The state space consists of 33 continuous states.

### Implementation of DDPG

The agent consists of an actor and a critic. Each containing a target and a local neural network.

The critic receives a state and an action as input and outputs the value it thinks that state-action pair is supposed to
have.

The actor received a state and outputs an action vector, each output node mapping to one continuous action.

#### Replay Buffer

After a state action reward, next_state pair was observed by the agent, it gets stored in the replay buffer, in order to
randomly sample these experiences after UPDATE_EVERY timestep.

#### Training

Because of its importance the function used for learning will be explained more in the following:

```python

states, actions, rewards, next_states, dones = experiences

# --- train the critic ---
next_actions = self.actor_target(next_states)
# Get expected Q values from local critic by passing in both the states and the actions
Q_expected = self.critic_local.forward(states, actions)
# Get next expected Q values from local critic by passing in both the next_states and the next_actions
next_Q_targets = self.critic_target(next_states, next_actions)
# Compute Q targets for current states
Q_targets = rewards + (gamma * next_Q_targets * (1 - dones))
# Caclulate the loss function using the expected return and the target
critic_loss = F.mse_loss(Q_expected, Q_targets)
self.critic_optim.zero_grad()
critic_loss.backward()
torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
self.critic_optim.step()

# --- train the actor ---
# create the action predictions by passing the states to the local network
actions_prediction = self.actor_local.forward(states)
# calculate the loss function of the actor
actor_loss = -self.critic_local(states, actions_prediction).mean()
self.actor_optim.zero_grad()
actor_loss.backward()
self.actor_optim.step()

# soft update the target networks
self.soft_updatee(self.critic_local, self.critic_target, TAU)
self.soft_updatee(self.actor_local, self.actor_target, TAU)
```

At first the states, actions, rewards, next states are extracted from the experiences.<br>

##### Training of the Critic:

The expected value of the current state is computed using the states and actions. (And called Q_expected)<br>
The expected value of the next state is computed using the next_states and next_actions. (And called next_Q_expected)<br>
The target value for the Q value is then calculated using the reward and Q_expected multiplied with discount factor gamma.
After having obtained all the values above it is possible to calculate the loss function and subsequently update the local network using gradient descent.

##### Training of the Actor
The prediction of the action taken is obtained by passing the current state through the local_actor network.
The result is used to compute the loss function, which is negated to perform gradient ascent on the state value.


#### Soft Updating

After this the target network is updated using the weights and biases from the trained local network and a factor Tau.
Tau determines how fast the target network should adopt the weights and biases from the local network.

Just copying the values of the local network to the target network was observed to be unstable.

### Model Architecture of Neural Model

#### Critic Structure
* The activation function chosen is RELU
* The first layer has the size of the state size in this case: 33
* The second layer has the size of the output of the first layer + the size of the action. In this case: 128 + 4
* The action is passed to the network in the second layer
* The third layer has a single output: representing the value of the given state

#### Actor Structure
* The activation function chosen for the first two layers is RELU
* The input size of the first layer is the size of the state size: in this case: 33
* The first layer has 128 nodes
* The second layer has 128 nodes
* The third layer has nodes corresponding to the action vector: in this case: 4
* The activation function for the last layer is tanh, in order to sqeeze the output space between -1 and 1


#### Optimizer

The chosen optimizer is Adam.

### Hyperparameters

#### Minibatch size

The minibatch size is the size of the batch used to learn by using gradient descent. <br>
One minibatch is sampled from the replay buffer.<br>
The chosen minibatch size is 128

### Gamma

Gamma is the discount factor.<br>
Ranging from 0 to 1 <br>
Bigger values of gamma correspond to a policy that values future reward just as much as immediate.<br>
The chosen value for gamma is 0.99

### Learning Rate Actor

The Learning rate for the actor optimizer is chosen to be: 0.00015

### Learning Rate Actor

The Learning rate for the actor optimizer is chosen to be: 0.00021

### Tau

Tau is used for soft update of target parameters. Larger values for tau were observed to have a hard time learning at
the beginning but perform better eventually.<br>
The selected value of Tau is 0.002

### Buffer Size

The constant BUFFER_SIZE determines the maximum amount of experiences that can be stored in the ReplayBuffer. <br>
The value chosen for this Project is 1000000

## Ideas for Future Work

### Recurrent Neural Networks
An a2c algorithm could be used to parallelize the training of multiple actors in multiple environments.




