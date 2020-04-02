from keras.models import Sequential, load_model
import numpy as np
import random 
from Grid import Grid, POSSIBLE_VALUES_IN_BOX
import os
from tqdm import tqdm

model = load_model("./models/c64xc64x64_5x5____19.56avg___27.00max____5.00min__1585846096.model")

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
        g.render(150)
    total.append(total_rewards)
print(f"Min: {min(total)}")
print(f"Max: {max(total)}")
print(f"Average: {sum(total) / 1000}")
