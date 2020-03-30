from keras.models import Sequential, load_model
import numpy as np
from Grid import Grid
import os

model = load_model("")

g = Grid()
end = False
total_rewards = 0
for i in range(1000):
    total_rewards = 0
    end = False
    state = g.reset()
    while not end:
        q_values = model.predict(np.array(state).reshape(-1, *state.shape)/2)[0]
        action = np.argmax(q_values)
        state, reward, end = g.step(action)
        total_rewards += reward
        g.render(50)
    print(total_rewards)
