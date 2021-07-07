############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import time
import collections
from collections import deque


class Agent:
    #set the class variable for action mapping: 0:right; 1:up; 2:left; 3:down
    DISCRETE_ACTIONS_MAPPING = {0:np.array([0.02, 0], dtype=np.float32),
                                1:np.array([0, 0.02], dtype=np.float32),
                                2:np.array([-0.02, 0], dtype=np.float32),
                                3:np.array([0, -0.02], dtype=np.float32)}
    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 750
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        #set the number of episodes taken
        self.episode_num = 0
        #set the number of step within one episode
        self.episode_steps = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        #initialise the dqn
        self.dqn = DQN()
        #initialise the experience replay buffer
        self.buffer = ReplayBuffer()
        #initialise the epsilon
        self.epsilon = 0.6
        #set the dacay rate
        self.decay_rate = 0.95
        #set the target network update frequency N
        self.N = 250
        #set the greedy policy check threshold, on an episode basis
        self.threshold = 8
        #the training flag
        self.training = True
        #store the last distance to goal of the previous greedy episode
        self.prev_greedy_dist_goal = float(1)
        #store the finishing time of the training
        self.end_time = time.time() + 600
    
    def epsilon_decay(self):
        self.epsilon = self.epsilon*self.decay_rate

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            if self.num_steps_taken != 0:
                self.epsilon_decay()  #decay the epsilon every episode
                #update the number of episodes done
                self.episode_num += 1
                #reset the episode step number
                self.episode_steps = 0
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        #action = np.random.uniform(low=-0.01, high=0.01, size=2).astype(np.float32)
        if self.num_steps_taken == 0:
            print(state)
        if self.has_check_greedy_reached():
            action = np.argmax(self.dqn.predict_single(state))
            #print(state)
        else:
            action = self.get_epsilon_greedy(state)
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        #must return a continous action to the env
        return Agent.DISCRETE_ACTIONS_MAPPING[action]
    
#    def reward_func(self, next_state, distance_to_goal):
#        base_reward = float(1 - distance_to_goal)
#        if (self.state[0] < 0.5):
#            if ((next_state == self.state).all()):
#                #less rewards for hitting a wall
#                reward = float(-0.05)
#            elif (next_state[0] < self.state[0]):
#                #less rewards for going backward
#                reward = float(-0.05)
#            elif (next_state[0] > self.state[0]):
#                #more rewards for going to the right, closer to target
#                reward = float(0.1)
#            else:
#                reward = float(0.01)
#        else:
#            if ((next_state == self.state).all()) and (self.state[0] < 0.9):
#                reward = base_reward - float(0.1)
#            elif (next_state[0] < self.state[0]) and (self.state[0] < 0.9):
#                reward = base_reward - float(0.1)
##            elif (next_state[0] > self.state[0]):
##                reward = base_reward
#            else:
#                reward = base_reward
#        return reward
    
    def reward_func(self, next_state, distance_to_goal):
        base_reward = float(1 - distance_to_goal)
        if (self.state[0] < 0.5):
            if ((next_state == self.state).all()):
                #less rewards for hitting a wall
                reward = float(-0.05)
            elif (next_state[0] < self.state[0]):
                #less rewards for going backward
                reward = float(-0.05)
            elif (next_state[0] > self.state[0]):
                #more rewards for going to the right, closer to target
                reward = float(0.2)
            else:
                reward = float(0.05)
        else:
            if ((next_state == self.state).all()) and (self.state[0] < 0.9):
                reward = base_reward - float(0.1)
            elif (next_state[0] < self.state[0]) and (self.state[0] < 0.9):
                reward = base_reward - float(0.1)
            elif (next_state[0] > self.state[0]) and (self.state[0] < 0.9):
                reward = base_reward - float(0.01)
            elif (next_state[0] == self.state[0]) and (self.state[0] < 0.9):
                reward = base_reward - float(0.05)
            else:
                reward = base_reward
        return reward

    
    def has_check_greedy_reached(self):
        if (self.episode_num+1)%self.threshold == 0:
            return True
        return False
        
    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        if self.has_check_greedy_reached():
            #checks for 100 steps of greedy policy, see if that already reaches target
            #print(next_state)
            self.episode_steps += 1
            if self.episode_steps == (self.episode_length-1):
                #check at the end of greedy episode
                if (self.state[0] < 0.9) and (distance_to_goal >= self.prev_greedy_dist_goal):
                    #reset the epsilon when the agent is stuck further away from the goal
                    if self.epsilon < 0.4:
                        self.epsilon = 0.5
#                    if (self.epsilon < 0.6) and (self.epsilon > 0.1):
#                        self.epsilon += 0.15
#                    elif self.epsilon <= 0.1:
#                        self.epsilon = 0.4
                elif (distance_to_goal < 0.4) or ((self.end_time - time.time()) < 300):
                    #increase check greedy frequency when target is close or the training has been going for a long time
                    self.threshold = 5
                elif distance_to_goal < 0.2:
                    self.threshold = 3
                #store the last distance to goal of the greedy episode
                self.prev_greedy_dist_goal = distance_to_goal
            
            if self.episode_steps <= 100:
                if distance_to_goal < 0.03:
                    #stops the training for the rest of the time when the greedy policy reaches the goal
                    self.training = False
            
            if (self.end_time - time.time()) <= 45:
                #risk management, to prevent sudden worsened greedy policy at the last minute
                if self.episode_steps <= 100:
                    if distance_to_goal < 0.5:
                        #for the last minute, stops the training when there is a satisfactory distance to goal
                        self.training = False

        else:
            if self.training:
                # Convert the distance to a reward
                #reward = float(1 - distance_to_goal)
                #or use a reward function
                reward = self.reward_func(next_state, distance_to_goal)
                # Create a transition
                transition = (self.state, self.action, reward, next_state)
                # Now you can do something with this transition ...
                #add transition to experience replay buffer
                self.buffer.add_transition(transition)
                if self.buffer.has_mb_reached():
                    #sample a minibatch for input to dqn
                    mb_transitions = self.buffer.sample_minibatch()
                    loss = self.dqn.train_q_network(mb_transitions)
                    #every N step updates the target network
                    if (self.num_steps_taken+1)%self.N == 0:
                        self.dqn.copy_weights_totarget()
        
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
            
        
    
    def get_epsilon_greedy(self, state):
        #action selection according to epsilon greedy at given state
        #get predicted Q values from dqn
        q_vals = self.dqn.predict_single(state)
        if np.random.uniform(0,1) < self.epsilon:
            action = np.random.choice([0,1,2,3])
        else:
            action = np.argmax(q_vals)
        return action

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        #action = np.array([0.02, 0.0], dtype=np.float32)
        #get predicted Q values from dqn
        q_vals = self.dqn.predict_single(state)
        #get the greedy action
        discrete_act = np.argmax(q_vals)
        #must return a continous action to the env
        return Agent.DISCRETE_ACTIONS_MAPPING[discrete_act]




# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        #Create the target network, same structure and initial weights as Q net
        self.target_network = Network(input_dimension=2, output_dimension=4)
        self.copy_weights_totarget()
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        #Define the discount factor
        self.gamma = 0.95
    
    def copy_weights_totarget(self):
        #copy weights from the Q network to the target network
        self.target_network.load_state_dict(self.q_network.state_dict(), strict=False)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition):
        #accept minibatch transitions
        #get all mb data
        mb_state_tensor = torch.tensor([transition[i][0] for i in range(len(transition))], dtype=torch.float32)
        mb_action_tensor = torch.tensor([transition[i][1] for i in range(len(transition))], dtype=torch.int64)  #int datatype for discrete action
        mb_reward_tensor = torch.tensor([transition[i][2] for i in range(len(transition))], dtype=torch.float32)
        mb_next_state_tensor = torch.tensor([transition[i][3] for i in range(len(transition))], dtype=torch.float32)
        
        #add data dimension to mb_action_tensor, for later second level indexing of the predictions
        mb_action_tensor = mb_action_tensor.unsqueeze(1)
        #get predictions(4 Q values for 4 actions for one state) as a mb tensor
        state_qvals_tensor = self.q_network(mb_state_tensor)
        #get the action specific prediction tensor for input into loss func
        state_action_qvals_tensor = state_qvals_tensor.gather(dim=1, index=mb_action_tensor).squeeze(1)
        
        #get next state action-state q values predictions
        #next_state_qvals_tensor = self.q_network(mb_next_state_tensor)
        next_state_qvals_tensor = self.target_network(mb_next_state_tensor).detach()  #detach the tensor from target network avoiding target network updating via loss
        #get second dimension maximum of next_state_qvals_tensor
        max_state_action_qvals_tensor,_ = next_state_qvals_tensor.max(dim=1)
        #compute the estimated target for the Bellman update
        target_tensor = mb_reward_tensor + self.gamma*max_state_action_qvals_tensor
        
        loss_func = torch.nn.MSELoss()
        #loss = loss_func(state_action_qvals_tensor, mb_reward_tensor)
        loss = loss_func(state_action_qvals_tensor, target_tensor)
        
        return loss
    
    
    def predict_single(self, input_state):
        #takes in an input state to trained dqn and predict the Q action values, returned as 1D np array
        state_tensor = torch.tensor(input_state).unsqueeze(0)
        predicted_q_vals = self.q_network(state_tensor)[0]
        #convert tensor to array
        predicted_q_vals = predicted_q_vals.detach().numpy()
        return predicted_q_vals




class ReplayBuffer:
    def __init__(self, max_capacity=5000, minibatch_size=100):
        self.buffer = collections.deque(maxlen=max_capacity)
        self.mb_size = minibatch_size
        self.max_size = max_capacity
    
    def add_transition(self, transition):
        self.buffer.append(transition)
        
    def sample_minibatch(self):
        mb_indices = np.random.choice(range(len(self.buffer)), self.mb_size)
        sample = []
        for idx in mb_indices:
            sample.append(self.buffer[idx])
        return sample
    
    def has_mb_reached(self):
        if len(self.buffer) >= self.mb_size:
            return True
        return False
