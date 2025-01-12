from tensorflow.keras.models import Sequential
from tensorflow.keras.models import *
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
import random
import pygame
import numpy as np
import pickle
import gym
import flappy_bird_gym
import time


gamma = 0.9  # Discount factor
epsilon = 0.8  # Initial exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay factor for exploration
learning_rate = 0.1
batch_size = 128
memory = []
max_memory = 20000  # Max size of memory
input_dim = 2
output_dim = 2

def create_q_network(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        #Dense(128, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer="adam", loss='mse')
    return model


q_network = create_q_network(input_dim, output_dim)
env = flappy_bird_gym.make("FlappyBird-v0")

def get_action(state, print_qvalue=False):
    if random.uniform(0, 1) < epsilon:
        return random.choices([0, 1], [3.5 / 4, 0.5 / 4], k=1)[0]  # Random action

    state = np.expand_dims(state, axis=0)
    q_values = q_network.predict(state, verbose=0)[0]
    if print_qvalue:
        print(f'Predicted: {q_values}')
    return np.argmax(q_values)


def play(count_plays, see_play=False, print_qvalues=False):
    global memory

    for iteration in range(count_plays):

        new_play = []
        env.reset()
        last_obs = np.zeros(input_dim)

        prev_score = 0
        while True:
            if see_play:
                env.render()
                time.sleep(1 / 30)

            action = get_action(last_obs, print_qvalues)

            obs, reward, terminated, info = env.step(action)

            score = info['score']
            reward = 0.1
            if terminated:
                reward = -1
            if score - prev_score > 0:
                reward = 1
            prev_score = score

            last_obs = obs

            if len(new_play) > 0:
                new_play[-1][3] = last_obs

            new_play.append([last_obs, action, reward, None])

            if score > 50:
                print("daaaaaaaa!!!")

            if terminated:
                print(score)
                break

        memory += new_play


def train_q_network():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)

    states, nxt_states = [], []
    for position in batch:
        states.append(position[0])

        if position[3] is None:
            nxt_states.append(np.zeros(input_dim))
        else:
            nxt_states.append(position[3])

    states = np.array(states)
    nxt_states = np.array(nxt_states)

    q_val_states = q_network.predict(states, verbose=0)
    q_val_nxt = q_network.predict(nxt_states, verbose=0)

    targets = []
    for id in range(batch_size):
        s, action, reward, nxt = batch[id]

        if nxt is None:
            targets.append(np.array([reward, reward]))

        else:
            target_f = q_val_states[id]
            target_f[action] = reward + gamma * np.max(q_val_nxt[id])
            targets.append(target_f)

    targets = np.array(targets)
    q_network.fit(states, targets, epochs=1, verbose=0)


def train(nr_interations, nr_trains):
    global epsilon
    global memory

    for i in range(nr_interations):
        print(f'Iteration: {i + 1}/{nr_interations}')

        for _ in range(nr_trains):
            train_q_network()

        play(1)

        if len(memory) > max_memory:
            random.shuffle(memory)
            memory = memory[:-(len(memory) - max_memory)]

        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay

        if (i+1) % 50 == 0:
            q_network.save('position_model_art2.keras')

    q_network.save('position_model_art2.keras')

#epsilon = 1
#play(200)
#epsilon = 0.8

#train(50000, 10)

#epsilon = epsilon_min
#play(100)
#q_network = load_model('position_model_150pipe.keras')


q_network = load_model('arhiva/position_model_art.keras')
epsilon = 0
play(10, True, False)