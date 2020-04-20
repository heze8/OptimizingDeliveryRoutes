import random
import numpy as np
from PIL import Image
import cv2
from RoadGeneration import generate_grid_with_roads, getFreePositions

# DIRECTIONS
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

# Grid Settings
SIZE = 5
RIDERS = 2 # MAXIMUM RIDERS IS 8.
DESTINATIONS = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_SPACE = len(ACTIONS) ** RIDERS

# Maximum number of steps before ending
MAX_STEPS = 100

# Objects in Grid
POSSIBLE_VALUES_IN_BOX = 3
ROAD_N = 0
DESTINATION_N = 1
RIDER_N = [2, 3, 4, 5, 6, 7, 8, 9]
UNPASSABLE_N = 10

COLOURS = { 0: (255, 255, 255),
         1: (0, 255, 255),
         2: (255, 255, 0),
         3: (255, 0, 0),
         4: (255, 125, 125),
         5: (255, 125, 0),
         6: (255, 0, 125),
         7: (255, 60, 180),
         8: (255, 180, 60),
         9: (255, 0, 255),
         10: (0, 0, 0)}

# REWARDS
# OOB = -5 # Rider goes out of bounds (i.e. unpassable terrain / out of grid)
OOB = -1 # Rider goes out of bounds (i.e. unpassable terrain / out of grid)
MAKE_DELIVERY = 10 # Rider successfully steps on box with destination
MOVE = -1 # Movement penalty, each rider will incur this penalty
MEET_OTHER_RIDER = -3 # Rider in same box as another rider, this encourages them to split up (?)
FAIL_IN_MAX_STEPS = 0 # Riders do not complete all deliveries in MAX_STEPS

"""
    UTILITY FUNCTIONS
"""
# Generated unique random tuples
def get_random_tuple(count, free_positions):
    unique_tuples = set()
    while len(unique_tuples) < count:
        tup = random.choice(free_positions)
        unique_tuples.add(tup)
    return unique_tuples

def generate_actions_dict(riders):
    action_dict = dict()
    for i in range(ACTION_SPACE):
        action_list = list()
        for j in range(riders):
            action_list.append(ACTIONS[(i // (len(ACTIONS)**(riders - j - 1)) % len(ACTIONS))])
        action_dict[i] = action_list
    return action_dict

def getLegalAction(riderPos, actions, grid):
    legal_actions = []
    for direction in actions:
        if direction == STAY: # No need to change rider position
            legal_actions.append(STAY)
            continue

        row = riderPos[0]
        col = riderPos[1]

        if direction == UP:
            row -= 1
        elif direction == DOWN:
            row += 1
        elif direction == LEFT:
            col -= 1
        elif direction == RIGHT:
            col += 1

        SIZE = len(grid)

        # Going out of boundaries
        if row < 0 or row > SIZE - 1 or col < 0 or col > SIZE - 1:
            continue
        elif grid[row][col] == UNPASSABLE_N: # Step on unpassable terrain
            continue
        elif grid[row][col] in RIDER_N: # Step on other rider
            continue

        legal_actions.append(direction)

    return legal_actions

"""
    GRID CLASS
"""
class Grid:
    def __init__(self):
        self.grid, self.rider_positions = self.initialize_grid()
        self.destinations = DESTINATIONS
        self.action_space = generate_actions_dict(RIDERS)
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
        # grid, free_positions = generate_grid_with_roads(SIZE, UNPASSABLE_N)
        grid = [[0 for i in range(SIZE)] for j in range(SIZE)]
        free_positions = getFreePositions(grid)
        positions = get_random_tuple(DESTINATIONS + 1, free_positions)
        rider_positions = list()
        position = positions.pop()
        for i in range(RIDERS):
            grid[position[0]][position[1]] = RIDER_N[i] # Assign number to matrix with last rider's index
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
    
                
            self.grid[self.rider_positions[i][0]][self.rider_positions[i][1]] = RIDER_N[i] # Set the updated rider position
        
        end = self.destinations == 0

        if not end and self.steps == MAX_STEPS: # If any delivery is not completed in max steps, then penalise
            reward += FAIL_IN_MAX_STEPS
            end = True

        return self.convert_grid_to_tensor(), reward, end

    def move(self, rider, direction):
        if direction == STAY: # No need to change rider position
            return False

        original_row = self.rider_positions[rider][0]
        original_col = self.rider_positions[rider][1]
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
        elif self.grid[row][col] == UNPASSABLE_N: # Step on unpassable terrain
            row = original_row
            col = original_col
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
        img = cv2.resize(np.array(img), (300, 300), interpolation=cv2.INTER_NEAREST)  
        cv2.imshow("image", np.array(img))  
        cv2.waitKey(delay)

    def get_image(self):
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8) 
        for i in range(SIZE):
            for j in range(SIZE):
                env[i][j] = COLOURS[self.grid[i][j]]
        img = Image.fromarray(env, 'RGB')
        return img
    
    def convert_grid_to_tensor(self):
        x = np.asarray(self.grid)
        x = x.reshape(SIZE, SIZE, 1)
        return x

# Random moving demo
# for i in range(10000):
#     g = Grid()
#     end = False
#     total_reward = 0
#     while not end:
#         action_n = random.randint(0, ACTION_SPACE-1)
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