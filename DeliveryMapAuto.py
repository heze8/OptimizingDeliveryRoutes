import random
import numpy as np
from PIL import Image
import cv2
from RoadGeneration import generate_grid_with_roads, getFreePositions
from astar import astar

class Rider:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Rider ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1
"""
    UTILITY FUNCTIONS
"""
# Generated unique random tuples
def get_random_tuple(count, free_positions):
    unique_tuples = set()
    while len(unique_tuples) < count:
        tup = random.choice(free_positions)
        free_positions.remove(tup)
        unique_tuples.add(tup)
    return unique_tuples
    
# def generate_actions_dict(riders):
#     action_dict = dict()
#     for i in range(NUM_ACTION):
#         action_list = list()
#         for j in range(riders):
#             action_list.append(ACTIONS[(i // (len(ACTIONS)**(riders - j - 1)) % len(ACTIONS))])
#         action_dict[i] = action_list
#     return action_dict


# DIRECTIONS
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

# Grid Settings
SIZE_MAP = 15
NUM_RIDER = 1 # MAXIMUM RIDERS IS 8.
NUM_DELIVERY = 10
# ACTIONS = [None, UP, DOWN, LEFT, RIGHT]
# NUM_ACTION = len(ACTIONS)

# Maximum number of steps before ending
MAX_STEPS = 100

# Objects in Grid
ROAD_N = 0
DESTINATION_N = 1
RIDER_N = [2, 3, 4, 5, 6, 7, 8, 9]
UNPASSABLE_N = -1

COLOURS = { ROAD_N: (255, 255, 255),
         DESTINATION_N: (0, 255, 255),
         2: (255, 255, 0),
         3: (255, 0, 0),
         4: (255, 125, 125),
         5: (255, 125, 0),
         6: (255, 0, 125),
         7: (255, 60, 180),
         8: (255, 180, 60),
         9: (255, 0, 255),
         UNPASSABLE_N: (0, 0, 0)}

# REWARDS
# OOB = -5 # Rider goes out of bounds (i.e. unpassable terrain / out of grid)
OOB = -3 # Rider goes out of bounds (i.e. unpassable terrain / out of grid)
MAKE_DELIVERY = 300 # Rider successfully steps on box with destination
MOVE = 1 # Movement penalty, each rider will incur this penalty
MEET_OTHER_RIDER = -3 # Rider in same box as another rider, this encourages them to split up (?)
FAIL_IN_MAX_STEPS = -10 # Riders do not complete all deliveries in MAX_STEPS
STAGNANT = -1

"""
    MultiAgentDeliveryEnv CLASS
"""
class MultiAgentDeliveryEnv:
    def __init__(self):
        self.destinationPos = []
        self.grid, self.rider_positions = self.initialize_grid()
        self.destinations = NUM_DELIVERY
        self.action_space = []
        self.observation_space = (SIZE_MAP, SIZE_MAP, 1)
        # self.action_space_size = NUM_ACTION
        self.steps = 0
    
    def reset(self):
        self.destinationPos = []
        self.grid, self.rider_positions = self.initialize_grid()
        self.destinations = NUM_DELIVERY
        self.steps = 0
        return self.returnStateInfo()

    # Initialise random positions for delivery
    # Initialise one random position for riders to start in
    # Delivery positions and rider positions guaranteed to not be in the same box
    def initialize_grid(self):
        grid, free_positions = generate_grid_with_roads(SIZE_MAP, UNPASSABLE_N)
        positions = get_random_tuple(NUM_DELIVERY + NUM_RIDER, free_positions)
        rider_positions = list()
        
        for i in range(NUM_RIDER):
            position = positions.pop()
            grid[position[0]][position[1]] = RIDER_N[i] # Assign number to matrix with last rider's index
            rider_positions.append(position)
        
        for i in range(NUM_DELIVERY):
            position = positions.pop()
            self.destinationPos.append(position)
            grid[position[0]][position[1]] = DESTINATION_N
        
        return grid, rider_positions

    # Execute action
    # Update grid and rider_positions
    # Return rewards and end?
    def step(self, action_n):
        destinationAction = self.destinationPos.pop(action_n)
        self.destinations = len(self.destinationPos)
    
        # First we "remove" the riders from the grid
        for i in range(NUM_RIDER):
            riderPos = self.rider_positions[i]
            self.grid[riderPos[0]][riderPos[1]] = ROAD_N

        
        
        reward = SIZE_MAP

        path = astar(self.grid, riderPos, destinationAction)
        if path == None:
            print(self.grid, riderPos, destinationAction)
            distance = 9999
        else:
            distance = len(path) - 1
        reward -= distance

        self.grid[destinationAction[0]][destinationAction[1]] = RIDER_N[i]
        self.rider_positions[i] = destinationAction

        self.steps += 1
           
        end = self.destinations == 0

        if self.steps % 2 == 0:
            newDes = get_random_tuple(1, getFreePositions(self.grid)).pop()
            self.destinationPos.append(newDes)
            self.grid[newDes[0]][newDes[1]] = 1

        return self.returnStateInfo(), reward, end
           
    def __str__(self):
        string = ""
        for row in range(SIZE_MAP):
            for col in range(SIZE_MAP):
                string += str(self.grid[row][col]) + " "
            string += "\n"
        return string
    
    # Displays the grid in a beautiful window
    def render(self, delay=1):
        img = self.get_image()
        img = cv2.resize(np.array(img), (500, 500), interpolation=cv2.INTER_NEAREST)  
        cv2.imshow("image", np.array(img))  
        cv2.waitKey(delay)

    def get_image(self):
        env = np.zeros((SIZE_MAP, SIZE_MAP, 3), dtype=np.uint8) 
        for i in range(SIZE_MAP):
            for j in range(SIZE_MAP):
                env[i][j] = COLOURS[self.grid[i][j]]
        img = Image.fromarray(env, 'RGB')
        return img
    
    def returnStateInfo(self):
        states = []
        for des in self.destinationPos:
            x = np.asarray(self.grid)
            x[des] = -2
            x = x.flatten()
            riderPos = self.rider_positions[0]
            path = astar(self.grid, riderPos, des)
            if path == None:
                print(self.grid, riderPos, des)
            distance = len(path) - 1
            x = np.append(x, distance)
            states.append(x)

        return states

# Random moving demo
# for i in range(10000):
#     g = Grid()
#     end = False
#     total_reward = 0
#     while not end:
#         action_n = random.randint(0, NUM_ACTION-1)
#         state, reward, end = g.step(action_n)
#         total_reward += reward
#         g.render(100)
#     print(total_reward)

# print(Grid())
# g = Grid()
# print(g)
# g.render(3000)
# g = Grid()
# g.render(3000)
# g = Grid()
# print(g)