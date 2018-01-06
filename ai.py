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