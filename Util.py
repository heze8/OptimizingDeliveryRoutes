import random
"""
    UTILITY FUNCTIONS
"""
# Generated unique random tuples
def get_random_tuple(count, size):
    unique_tuples = set()
    while len(unique_tuples) < count:
        row = random.randint(0, size - 1)
        col = random.randint(0, size - 1)
        unique_tuples.add((row, col))
    return unique_tuples

def generate_actions_dict(riders, actions):
    action_dict = dict()
    action_space = 4**riders
    for i in range(action_space):
        action_list = list()
        for j in range(riders):
            action_list.append(actions[(i // (4**(riders - j - 1)) % 4)])
        action_dict[i] = action_list
    return action_dict