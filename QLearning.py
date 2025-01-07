from tensorflow.keras.models import Sequential
from tensorflow.keras.models import *
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
import gymnasium as gym
import flappy_bird_gymnasium
import random
import pygame
import numpy as np

gamma = 0.9         # Discount factor
epsilon = 0.8        # Initial exploration rate
epsilon_min = 0.1    # Minimum exploration rate
epsilon_decay = 0.995  # Decay factor for exploration
learning_rate = 0.1
batch_size = 8
memory = []
max_memory = 2000    # Max size of memory
input_dim = 180
output_dim = 2

def create_q_network(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(16, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

q_network = create_q_network(input_dim, output_dim)


def get_action(state, print_qvalue=False):
    if random.uniform(0, 1) < epsilon:
        return random.choices([0,1], [10/20, 10/20], k=1)[0]  # Random action

    state = np.expand_dims(state, axis=0)
    q_values = q_network.predict(state, verbose = 0)
    if print_qvalue:
        print(f'Predicted: {q_values}')
    return np.argmax(q_values)


def play(count_plays, see_play = False, print_qvalues = False):
    count_frame = 0
    if see_play:
        screen = pygame.display.set_mode((288, 512))

    env = None
    if see_play:
        env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    else:
        env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=True)
    

    for iteration in range(count_plays):
        new_play = []
        env.reset()
        last_obs = np.zeros(input_dim)

        while True:
            count_frame = count_frame + 1
            action = get_action(last_obs)

            obs, reward, terminated, _, info = env.step(action)
            last_obs = obs
            
            new_play.append((last_obs, action, reward))

            if terminated:
                memory.append(new_play)
                break

    env.close()


def train_q_network(nr_interations):
    if len(memory) < batch_size:
        return

    states = []
    targets = []

    for iteration in range(nr_interations):
        batch = random.sample(memory, batch_size)
        for play_id in range(batch_size):
            states_q = []
            for state_id in range(len(batch[play_id])):
                states_q.append(batch[play_id][state_id][0])

            states_q = np.array(states_q)
            q_values = q_network.predict(states_q, verbose=0)

            for state_id in range(len(batch[play_id])):

                state, action, reward = batch[play_id][state_id]

                if state_id == len(batch[play_id]) - 1:
                    states.append(state)
                    targets.append(np.array([reward, reward]))
                else:
                    target = reward + gamma * np.max(q_values[state_id + 1])
                    target_f = q_values[state_id]
                    target_f[action] = target
                    states.append(state)
                    targets.append(target_f)

        print(iteration)
        q_network.fit(np.array(states), np.array(targets), epochs=1, verbose=1)

def train(nr_interations, epochs):
    global epsilon
    global memory

    for i in range(nr_interations):
        print(f'Iteration: {i+1}/{nr_interations}')
        print(f'epsilon: {epsilon}')

        train_q_network(epochs)
        play(10)

        if len(memory) > max_memory:
            random.shuffle(memory)
            memory = memory[:-10]

        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay

        print(f'epsilon: {epsilon}')
        
        if i%10 == 0:
            q_network.save("lidar_model.keras")



play(50)
train(200, 1)
epsilon = 0
play(5, True)
