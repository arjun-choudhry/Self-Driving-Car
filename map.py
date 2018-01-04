# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5,3,0.9)
action2rotation = [0,20,-20]
last_reward = 0
scores = []

# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    goal_x = 20
    goal_y = largeur - 20
    first_update = False

# Initializing the last distance
last_distance = 0

# Initializing the map
first_update = True
def init():
    # sand variable will be an array where the cells will be the pixels of the map and each cell will have 1 if there is sand and 0 if no sand. In the beginning, each sand will be 0.
    global sand
    # goal is the point in the map where we train the car to go. Initially, we have set the goal as the upper left corner of the map. Once, it reaches that goal, we change the goal to bottom right, and then we reverse the decision.
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    goal_x = 20
    goal_y = largeur - 20
    first_update = False

# Initializing the last distance
# last_distance gives the current distance of the car to the goal
last_distance = 0

# Creating the car class

class Car(Widget):
    # angle is the angle between the x-axis and the axis along which the car is currently moving
    angle = NumericProperty(0)
    # rotation remembers the car's last rotation, ie 0deg, 20deg or -20deg
    rotation = NumericProperty(0)
    # velocity vectors and the vector of coordinates of velocity
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    # Car that we make will have 3 sensors as follows:
    # 1st sensor: Detects any sand infront of the car
    # 2nd sensor: Detects any sand left of the car
    # 3rd sensor: Detects any sand right of the car
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    # Signal vectors are the signals received from each of the sensors. These signals measure the density of sand around the respective sensors. We calculate the density of sand by taking 20 by 20 box around the sensor. In this box, we divide the number of 1's in the box by the total number of boxes in that area(400).
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)