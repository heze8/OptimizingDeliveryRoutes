from keras.models import Sequential, load_model
import numpy as np
from Grid import Grid, POSSIBLE_VALUES_IN_BOX
import os

model = load_model("./models/c32x32_5x5___-95.85avg___27.00max_-152.00min__1585820841.model")

g = Grid()
end = False
total_rewards = 0
for i in range(100):
    total_rewards = 0
    end = False
    state = g.reset()
    while not end:
        q_values = model.predict(np.array(state).reshape(-1, *state.shape)/POSSIBLE_VALUES_IN_BOX)[0]
        action = np.argmax(q_values)
        state, reward, end = g.step(action)
        total_rewards += reward
        g.render(10)
    print(total_rewards)
