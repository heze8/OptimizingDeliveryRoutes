import random
import numpy as np
from PIL import Image
import cv2
import Util as util

# DIRECTIONS
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

# Grid Settings
SIZE = 7
RIDERS = 3
DESTINATIONS = 5
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_SPACE = len(ACTIONS) ** RIDERS

# Maximum number of steps before ending
MAX_STEPS = 35

# Objects in Grid
RIDER_N = 2
DESTINATION_N = 1
ROAD_N = 0
COLOURS = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

# REWARDS
OOB = -10 # Rider goes out of bounds (i.e. unpassable terrain / out of grid)
MAKE_DELIVERY = 100 # Rider successfully steps on box with destination
MOVE = -3 # Movement penalty, each rider will incur this penalty
MEET_OTHER_RIDER = -3 # Rider in same box as another rider, this encourages them to split up (?)
FAIL_IN_MAX_STEPS = -200 # Riders do not complete all deliveries in MAX_STEPS

"""
    GRID CLASS
"""
class Grid:
    def __init__(self):
        self.grid, self.rider_positions = self.initialize_grid()
        self.destinations = DESTINATIONS
        self.action_space = util.generate_actions_dict(RIDERS, actions)
        self.observation_space = (SIZE, SIZE, 1)
        self.action_space_size = ACTION_SPACE
        self.steps = 0
    
    def reset(self):
        self.grid, self.rider_positions = self.initialize_grid()
        self.destinations = DESTINATIONS
        self.steps = 0
        return self.convert_grid_to_tensor()

    # Initialise random positions for delivery
    # Initialise one random position for riders to start in
    # Delivery positions and rider positions guaranteed to not be in the same box
    def initialize_grid(self):
        grid = [[ROAD_N for i in range(SIZE)] for i in range(SIZE)]
        positions = util.get_random_tuple(DESTINATIONS + 1, SIZE)
        rider_positions = list()
        position = positions.pop()
        grid[position[0]][position[1]] = RIDER_N
        for i in range(RIDERS):
            rider_positions.append(position)
        
        for i in range(DESTINATIONS):
            position = positions.pop()
            grid[position[0]][position[1]] = DESTINATION_N
        
        return grid, rider_positions

    # Execute action
    # Update grid and rider_positions
    # Return rewards and end?
    def step(self, action_n):
        self.steps += 1
        # First we "remove" the riders from the grid
        for i in range(RIDERS):
            self.grid[self.rider_positions[i][0]][self.rider_positions[i][1]] = ROAD_N

        action = self.action_space[action_n]
        reward = 0
        # Check if out of bounds or reached destination
        for i in range(RIDERS):
            if self.move(i, action[i]):
                reward += OOB
            else:
                if self.grid[self.rider_positions[i][0]][self.rider_positions[i][1]] == DESTINATION_N:
                    reward += MAKE_DELIVERY
                    self.destinations -= 1
                else: 
                    if action[i] != STAY: # Don't penalise if rider chooses to stay
                        reward += MOVE
                
                # if self.grid[self.rider_positions[i][0]][self.rider_positions[i][1]] == RIDER_N: # Penalise if rider meets another rider (encourage them to separate)
                #     reward += MEET_OTHER_RIDER
    
                
            self.grid[self.rider_positions[i][0]][self.rider_positions[i][1]] = RIDER_N # Set the updated rider position
        
        end = self.destinations == 0

        if not end and self.steps > MAX_STEPS: # If any delivery is not completed in max steps, then penalise
            reward += FAIL_IN_MAX_STEPS
            end = True

        return self.convert_grid_to_tensor(), reward, end

    def move(self, rider, direction):
        if direction == STAY: # No need to change rider position
            return False

        row = self.rider_positions[rider][0]
        col = self.rider_positions[rider][1]
        if direction == UP:
            row -= 1
        elif direction == DOWN:
            row += 1
        elif direction == LEFT:
            col -= 1
        elif direction == RIGHT:
            col += 1

        oob = False
        # Going out of boundaries
        if row < 0: 
            row = 0
            oob = True
        elif row > SIZE - 1:
            row = SIZE - 1
            oob = True
        elif col < 0:
            col = 0
            oob = True
        elif col > SIZE - 1:
            col = SIZE - 1
            oob = True
        
        self.rider_positions[rider] = (row, col)
        return oob
            
    def __str__(self):
        string = ""
        for row in range(SIZE):
            for col in range(SIZE):
                string += str(self.grid[row][col]) + " "
            string += "\n"
        return string
    
    # Displays the grid in a beautiful window
    def render(self, delay=1):
        img = self.get_image()
        img = img.resize((300, 300))  
        cv2.imshow("image", np.array(img))  
        cv2.waitKey(delay)

    def get_image(self):
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        for i in range(SIZE):
            for j in range(SIZE):
                if self.grid[i][j] == DESTINATION_N:
                    env[i][j] = COLOURS[DESTINATION_N]
                elif self.grid[i][j] == RIDER_N:
                    env[i][j] = COLOURS[RIDER_N]
        img = Image.fromarray(env, 'RGB')
        return img
    
    def convert_grid_to_tensor(self):
        x = np.asarray(self.grid)
        x = x.reshape(SIZE, SIZE, 1)
        return x

# Random moving demo

# g = Grid()
# end = False
# total_reward = 0
# while not end:
#     action_n = random.randint(0, ACTION_SPACE-1)
#     state, reward, end = g.step(action_n)
#     total_reward += reward
#     g.render(50)
# print(total_reward)
