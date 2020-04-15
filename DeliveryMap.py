import random
import numpy as np
from PIL import Image
import cv2
from RoadGeneration import generate_grid_with_roads, getFreePositions

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
    
def generate_actions_dict(riders):
    action_dict = dict()
    for i in range(NUM_ACTION):
        action_list = list()
        for j in range(riders):
            action_list.append(ACTIONS[(i // (len(ACTIONS)**(riders - j - 1)) % len(ACTIONS))])
        action_dict[i] = action_list
    return action_dict


# DIRECTIONS
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
STAY = 0

# Grid Settings
SIZE_MAP = 10
NUM_RIDER = 1 # MAXIMUM RIDERS IS 8.
NUM_DELIVERY = 3
ACTIONS = [STAY, UP, DOWN, LEFT, RIGHT]
NUM_ACTION = len(ACTIONS)

# Maximum number of steps before ending
MAX_STEPS = 100

# Objects in Grid
POSSIBLE_VALUES_IN_BOX = 3
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
MAKE_DELIVERY = 130 # Rider successfully steps on box with destination
MOVE = -1 # Movement penalty, each rider will incur this penalty
MEET_OTHER_RIDER = -3 # Rider in same box as another rider, this encourages them to split up (?)
FAIL_IN_MAX_STEPS = -10 # Riders do not complete all deliveries in MAX_STEPS
STAGNANT = -1

"""
    MultiAgentDeliveryEnv CLASS
"""
class MultiAgentDeliveryEnv:
    def __init__(self):
        self.grid, self.rider_positions = self.initialize_grid()
        self.destinations = NUM_DELIVERY
        self.action_space = ACTIONS
        self.observation_space = (SIZE_MAP, SIZE_MAP, 1)
        self.action_space_size = NUM_ACTION
        self.steps = 0
    
    def reset(self):
        self.grid, self.rider_positions = self.initialize_grid()
        self.destinations = NUM_DELIVERY
        self.steps = 0
        return self.convert_grid_to_tensor()

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
            grid[position[0]][position[1]] = DESTINATION_N
        
        return grid, rider_positions

    # Execute action
    # Update grid and rider_positions
    # Return rewards and end?
    def step(self, action_n):
        self.steps += 1
        # First we "remove" the riders from the grid
        for i in range(NUM_RIDER):
            self.grid[self.rider_positions[i][0]][self.rider_positions[i][1]] = ROAD_N

        action = self.action_space[action_n]
        reward = 0
        # Check if out of bounds or reached destination
        for i in range(NUM_RIDER): # doesnt this mean it controls all the riders together
            if self.move(i, action):
                reward += OOB
            elif self.grid[self.rider_positions[i][0]][self.rider_positions[i][1]] == DESTINATION_N:
                reward += MAKE_DELIVERY
                self.destinations -= 1
            elif action != STAY: 
                reward += MOVE
            elif action == STAY:
                reward += STAGNANT


                
                # if self.grid[self.rider_positions[i][0]][self.rider_positions[i][1]] == RIDER_N: # Penalise if rider meets another rider (encourage them to separate)
                #     reward += MEET_OTHER_RIDER
    
                
            self.grid[self.rider_positions[i][0]][self.rider_positions[i][1]] = RIDER_N[i] # Set the updated rider position
        
        end = self.destinations == 0

        if not end and self.steps == MAX_STEPS: # If any delivery is not completed in max steps, then penalise
            reward += FAIL_IN_MAX_STEPS
            end = True

        return self.convert_grid_to_tensor(), reward, end

    #ToDO move reward here
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
        elif row > SIZE_MAP - 1:
            row = SIZE_MAP - 1
            oob = True
        elif col < 0:
            col = 0
            oob = True
        elif col > SIZE_MAP - 1:
            col = SIZE_MAP - 1
            oob = True
        elif self.grid[row][col] == UNPASSABLE_N: # Step on unpassable terrain
            row = original_row
            col = original_col
            oob = True
        
        self.rider_positions[rider] = (row, col)
        return oob
            
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
    
    def convert_grid_to_tensor(self):
        x = np.asarray(self.grid)
        x = x.reshape(SIZE_MAP, SIZE_MAP, 1)
        return x

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