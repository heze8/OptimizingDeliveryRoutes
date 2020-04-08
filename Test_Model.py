from keras.models import Sequential, load_model
import numpy as np
import random 
from Grid import Grid, POSSIBLE_VALUES_IN_BOX
import os
from tqdm import tqdm
import sys

model = load_model("./models/c64xc64x64___-62.56avg___15.00max_-189.00min__1586315279.model")

random.seed(1)

g = Grid()
end = False
total = []
for i in tqdm(range(1000)):
    total_rewards = 0
    end = False
    state = g.reset()
    while not end:
        q_values = model.predict(np.array(state).reshape(-1, *state.shape)/POSSIBLE_VALUES_IN_BOX)[0]
        action = np.argmax(q_values)
        state, reward, end = g.step(action)
        total_rewards += reward
        g.render(5)#int(sys.argv[1]))
    total.append(total_rewards)
print(f"Min: {min(total)}")
print(f"Max: {max(total)}")
print(f"Average: {sum(total) / 1000}")
