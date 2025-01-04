from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import gymnasium as gym
import flappy_bird_gymnasium
import random
import numpy as np
from torch.backends.mkl import verbose

from UseEncoder import *
import pygame
import time
import keyboard

def create_q_network(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(output_dim, activation='linear')  # Output Q-values for each action
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# Hyperparameters
gamma = 0.9         # Discount factor
epsilon = 1        # Initial exploration rate
epsilon_min = 0.1    # Minimum exploration rate
epsilon_decay = 0.995  # Decay factor for exploration
learning_rate = 0.001
batch_size = 4
memory = []          # Replay memory for experience replay
max_memory = 2000    # Max size of memory
input_dim = 2304    # Encoded state size (~1000 features)
output_dim = 2       # Actions: [Flap, Do Nothing]

q_network = create_q_network(input_dim, output_dim)

def get_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choices([0,1], [19/20, 1/20], k=1)[0]  # Random action
    q_values = q_network.predict(state, verbose = 0)
    return np.argmax(q_values)

def play_by_hand(count_plays):
    screen = pygame.display.set_mode((288, 512))
    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)

    for iteration in range(count_plays):
        new_play = []
        env.reset()

        while True:
            image = env.render()
            encoded_image = encode_image(image)
            encoded_image = np.expand_dims(encoded_image, axis=0)

            action = 0
            if keyboard.is_pressed('space'):
                action = 1

            surface = pygame.surfarray.make_surface(image.transpose(1, 0, 2))
            screen.blit(surface, (0, 0))
            pygame.display.flip()


            obs, reward, terminated, _, info = env.step(action)

            if reward > 0:
                reward += 10  # bonus for passing pipes
            elif terminated:  # the bird hits the ground or a pipe
                reward -= 20  # penalize heavily for losing

            new_play.append((encoded_image, action, reward))

            if terminated:
                memory.append(new_play)
                break

    env.close()


def play(count_plays, see_play = False):
    if see_play:
        screen = pygame.display.set_mode((288, 512))
    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)

    for iteration in range(count_plays):
        new_play = []
        env.reset()

        while True:
            image = env.render()
            encoded_image = encode_image(image)
            encoded_image = np.expand_dims(encoded_image, axis=0)

            action = get_action(encoded_image)

            if see_play:
                surface = pygame.surfarray.make_surface(image.transpose(1, 0, 2))
                screen.blit(surface, (0, 0))
                pygame.display.flip()

            obs, reward, terminated, _, info = env.step(action)

            if reward > 0:
                reward += 10  # bonus for passing pipes
            elif terminated:  # the bird hits the ground or a pipe
                reward -= 20  # penalize heavily for losing

            new_play.append((encoded_image, action, reward))

            if terminated:
                memory.append(new_play)
                break

    env.close()


def train_q_network():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    for play_id in range(batch_size):
        for state_id in range(len(batch[play_id])):

            state, action, reward = batch[play_id][state_id]

            if state_id == len(batch[play_id]) - 1:
                q_network.fit(state, np.array([[reward, reward]]), epochs=1, verbose=0)
            else:
                next_state = batch[play_id][state_id + 1][0]
                target = reward + gamma * np.max(q_network.predict(next_state))
                target_f = q_network.predict(state)
                target_f[0][action] = target
                q_network.fit(state, target_f, epochs=1, verbose=0)

def train():
    global epsilon

    #play(5,True)
    play_by_hand(16)
    for i in range(4):
        train_q_network()
    epsilon = 0
    memory.clear()
    play(10,True)

train()