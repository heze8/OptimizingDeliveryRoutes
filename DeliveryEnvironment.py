import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time

SIZE = 10
MOVE_REWARD = -1
DELIVERY_REWARD = 5
NUMBER_DELIVERY_LOCATIONS = 2

RIDER = 1
DELIVERY_LOCATION = 2
DIVIDER = 3

COLOURS =  {1: (255, 175, 0), 2: (0, 255, 0), 3: (255, 255, 255)}

class Delivery_Environment:
    def __init__(self):
        self.rider = self.Blob()
        self.delivery_locations = list()
        self.size = 10
        exclude_set = set([self.rider.get_tuple_location()])
        for i in range(NUMBER_DELIVERY_LOCATIONS):
            delivery_location = self.Blob(exclude_set)
            self.delivery_locations.append(delivery_location)
            exclude_set.add(delivery_location.get_tuple_location())
    
    def get_state(self):
        return self.rider, self.delivery_locations

    def action(self, choice):
        self.rider.action(choice)
        for delivery_location in self.delivery_locations:
            if self.rider.same_position_as(delivery_location):
                reward = DELIVERY_LOCATION
                self.delivery_locations.remove(delivery_location)
            else: 
                reward = MOVE_REWARD
        
        done = False
        if len(self.delivery_locations) == 0:
            done = True

        return self.rider, self.delivery_locations, done, reward
    
    def render(self):
        env = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)  # starts an rbg of our size
        env[self.rider.x][self.rider.y] = COLOURS[RIDER]
        for delivery_location in self.delivery_locations:
            env[delivery_location.x][delivery_location.y] = COLOURS[DELIVERY_LOCATION]

        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        img = img.resize((600, 600))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        if cv2.waitKey(500) & 0xFF == ord('q'):
            pass
    
    def reset(self):
        return Delivery_Environment()
    
    class Blob:
        def __init__(self, exclude=None):
            position = (np.random.randint(0, SIZE), np.random.randint(0, SIZE))
            if exclude:
                while position in exclude:
                    position = (np.random.randint(0, SIZE), np.random.randint(0, SIZE))
            self.x = position[0]
            self.y = position[1]
        
        def get_tuple_location(self):
            return (self.x, self.y)

        def __str__(self):
            return f"{self.x}, {self.y}"

        def __sub__(self, other):
            return (self.x-other.x, self.y-other.y)
        
        def same_position_as(self, other):
            return self.x == other.x and self.y == other.y

        def action(self, choice):
            '''
            Gives us 4 total movement options. (0,1,2,3)
            '''
            if choice == 0:
                self.move(x=0, y=1)
            elif choice == 1:
                self.move(x=0, y=-1)
            elif choice == 2:
                self.move(x=1, y=0)
            elif choice == 3:
                self.move(x=-1, y=0)

        def move(self, x=False, y=False):
            # If we are out of bounds, fix!
            if self.x < 0:
                self.x = 0
            elif self.x > SIZE-1:
                self.x = SIZE-1
            if self.y < 0:
                self.y = 0
            elif self.y > SIZE-1:
                self.y = SIZE-1
    