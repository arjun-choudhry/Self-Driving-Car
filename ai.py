# -*- coding: utf-8 -*-
"""
"""
# AI file to choose the right action at the right time

# ******************************************
# Importing the libraries
# ******************************************
import numpy as np
import random
import os # useful when we need to load the model
import torch
import torch.nn as nn
import torch.nn.functional as fnc
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# ******************************************
# Creating the architecture of the Neural Network
# ******************************************
class Network(nn.Module):
    # output_neurons corresponds to the number of actions, and we have 3 possible actions, left,straight,right
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        # Specifying the input layer and the output neurons and attaching them to the object
        self.input_size = input_size
        self.nb_action = nb_action
        #Specifying the full connections existing between the layers of our neural networks. We will create the neural network containing only 1 hidden layer and hence the total number of connections will be 2(1 btw the input layer and the hidden layer and the other btw the hidden layer and the output layer). By full connections we mean that all the neurons of the input layer will be connected to all the neurons of the hidden layer.
        self.fc1 = nn.Linear(input_size, 30)
        # Establishing connection between the hidden layer and the output layer
        self.fc2 = nn.Linear(30, nb_action)

        # The below function is the function that will perform forward propagation and will activate the neurons and will return the Q-values corresponding to each action
    def forward(self,state):
        x = fnc.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# ******************************************
# Implementing Experience Replay
# ******************************************
""" There will be 3 functions in this class:
        1st will be the init function that will be used to define the variables that will be attached to the objects that will be instantiated from this class. The variables will be 'memory' of the 100 events and the 'capacity' which states the number of events that can be stored in the memory(in this examples, 100).
        2nd will be the push function that will push the events in the memory and ensure that the memory never contains more events than its capacity
        3rd will be the sample function that will be used to sample a few random events from the last 100 events that is stored"""
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        # memory contains the last 100 events
        self.memory = []

    # The event that we are appending in the memory is an array of 4 things: last_state, new_state, last_action, last_reward
    def push(self, event):
        self.memory.append(event)
        if(self.memory.length > self.capacity):
            del self.memory[0]
    # the 2nd argument is the batch size that states how many samples need to be present in each batch for the agent to learn from

""" Using the random.sample function, we are taking random samples of size 'batch_size' from the 'self.memory' element.
    zip(*) function reshapes the array like this. eg, we have an array like the following: list = [[1,2,3],[4,5,6]], then zip(*) will reshape the list as follows: [[1,4],[2,5],[3,6]]. We need to reshape the arrays because each of the entry in the memory will be of the form [last_state, new_state, last_action, last_reward]. However, for the learning process, we require the information clustered together in groups belonging to the same category. Hence, last_state will be one array, new_state will be one array and so on. Hence after the zip(*) is applied, we will have the following structure of sample: sample -> [array of last_state, array of new_state, array of last_action, array of last_reward] """
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory,batch_size))
        # Mapping the samples to the pytorch variable
        samples = map(lambda x: Variable(torch.cat(x,0)),samples)
        return samples

# ******************************************
# Implementing Deep Q Learning
# ******************************************
""" The deep q learning class will implement the deep q learning algorithm. It will internally use the 2 classes created above. It will contain the following functions:
init() -> This will be used to initialize the variables attached to the object that will be created from this class.
select_action() will be there to select the correct action for a particular time.
We will also have an update(), score() to update the score and see how the learning is going on, save() to save the brain of the car and the load() to load the brain"""
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size,nb_action)
        self.memory = ReplayMemory(100000)
        # Now, we will create an object of the optimizer. We are choosing the Adam optimizer.
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # For pytorch, we need the vector to be a torch tensor as well as it needs to have one more dimension, which corresponds to the batch as we want the last_state to be the input to the neural network, but when working with neural networks in general, whether with pytorch, tensorflow or keras, input vectors cannot be a simple vector by itself, it has to be in a batch. The network can only accept a batch of input vectors. Hence, we will not only create a tensor but also attach a batch parameter. Hence, we need to create a fake dimension corresponding the the batch, and this fake dimension will be the 1st dimension
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    # In this function, we will input the state to the neural network, get the outputs as the q-values and then apply these q-values to the softmax function to finally get the desired action.
    def select_action(self, input_state):
        probs = fnc.softmax(self.model(Variable(input_state, volatile = True))*10) # T=7
        action = probs.multinomial()
        # The below will return either 0,1,2 corresponding to the actions
        return action.data[0,0]

    # In the below function, we will train the deep q learning network, ie we will do the whole process of forward-propagation, back-propagation, ie we will get the output, compare the output with the target to compute the loss error, then backpropagate this loss error into the neural network and then using the sgd, we will adjust the weights depending on how much they contributed to the loss parameter.
    def learn(self,batch_state, batch_next_state, batc_reward, batch_action):
        # We want the model outputs of the input_states of our batch inputs.
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    # The following function updates the score and see how the learning is going on
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)

    # The below function saves the learnings of the agent in the memory
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    }, 'last_brain.pth')

    # The below function loads the learnings of the agent from the memory
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("checkpoint loaded !")
        else:
            print("no checkpoint found!")