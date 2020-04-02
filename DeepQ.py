from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
from Grid import Grid, SIZE, POSSIBLE_VALUES_IN_BOX
from tqdm import tqdm

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf
import numpy as np
import random
import time
import os

# Q Learning settings 
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 10000 # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000 # Minimum number of steps in a memory to start training
MODEL_NAME = f"c32x32_{SIZE}x{SIZE}" 
MINIBATCH_SIZE = 64 # How many steps (samples) to use for training
MIN_REWARD = -250 # FOR MODEL SAVE
UPDATE_TARGET_EVERY = 5 # Terminal states (end of episodes)

# Neural network settings
LEARNING_RATE = 0.001

LOAD_MODEL = None
# Environment settings
EPISODES = 20000

# Exploration settings
epsilon = 1 # not a constant, going to be decayed
EPSILON_DECAY = 0.99979
MIN_EPSILON = 0.001

# Stats settings
AGGREGATE_STATS_EVERY = 100 # Episodes
SHOW_PREVIEW = False

# If don't want to use GPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# GPU settings
# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# random.seed(1)
# np.random.seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

# Own Tensorboard class
# Override Tensorboard to create less log files as we are going to .fit() a lot of times
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        # self.log_dir = ".\\logs" # For windows
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass
    
    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

class DQNAgent:
    def __init__(self, env):
        # Main Model .train every step
        self.env = env
        self.model = self.create_model()

        # Target Model .predict every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0 # When to update model

    def create_model(self):
        if LOAD_MODEL is not None:
            model = load_model(LOAD_MODEL)

        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=self.env.observation_space))
        model.add(Activation("relu"))
        # model.add(Dropout(0.1))

        model.add(Flatten())
        # model.add(Dense(64, activation="relu")) 
        model.add(Dense(32, activation="relu")) 

        model.add(Dense(self.env.action_space_size, activation = "linear"))
        model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/POSSIBLE_VALUES_IN_BOX)[0]
    
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        current_states = np.array([transition[0] for transition in minibatch])/POSSIBLE_VALUES_IN_BOX
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])/POSSIBLE_VALUES_IN_BOX
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done: 
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
        
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)
        
        self.model.fit(np.array(X)/POSSIBLE_VALUES_IN_BOX, np.array(y), batch_size = MINIBATCH_SIZE, verbose=0, shuffle=False) #, callbacks=[self.tensorboard] if terminal_state else None)

        # updating to determine if we want to update target_model
        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

agent = DQNAgent(Grid())

# For Stats
ep_rewards = [MIN_REWARD]

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episode"):
    agent.tensorboard.step = episode

    episode_reward = 0 
    step = 1
    current_state = agent.env.reset()

    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, agent.env.action_space_size)
        
        new_state, reward, done = agent.env.step(action)

        episode_reward += reward
        
        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            agent.env.render()
        
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if average_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{average_reward:_>7.2f}avg_{max_reward:_>7.2f}max_{min_reward:_>7.2f}min__{int(time.time())}.model')
        
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
