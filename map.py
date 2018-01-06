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

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

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

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.

        # Writing the below code to not allow the car to go outside the maze defined. We make the density of the sand signal as 1,ie full sand. When it gets too close to the wall, then we punish the car by giving it a punishment
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class
class Game(Widget):
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        # last_signal is an array that containes the signals returned from the 3 sensors as well as the orientation parameters. 'orientation' stands for the orientation of the car w.r.t goal. Hence, if it is heading towards the goal, it will have a value of 0. If it is travelling slightly towards right, then the orientation will have a value of 45deg and so on. By adding the -orientaion parameter, we ensure that the car is exploring in the other direction as well and not only in the same direction. These 5 parameters are the input vectors that go into the network and these 5 inputs are adjusted by the weights.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        # The action below is the brain of our AI. It contains the output of the neural network result returned by the brain.update function. This function is present in the ai.py class. It takes 2 inputs as parameters, the last_signal array that we prepared in the previous step and the last_reward input.
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        # We update the rotation vector and move the car in the requisite direction
        rotation = action2rotation[action]
        self.car.move(rotation)
        # We update the distance of the car from the goal
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        # If the car goes in the sand, we slow down the spped and give it a reward of -1
        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1

        # If the car is closer to the goal, we give a positive reward else we it a slightly negative reward
        else: # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        # Setting the penalty for getting closer to the boundaries
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        # Updating the goal once the car reaches the goal
        if distance < 100:
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y

        last_distance = distance


# Adding the painting tools
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)
class CarApp(App):
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()

