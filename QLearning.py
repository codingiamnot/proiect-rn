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
epsilon = 0.5        # Initial exploration rate
epsilon_min = 0.1    # Minimum exploration rate
epsilon_decay = 0.99  # Decay factor for exploration
learning_rate = 0.1
batch_size = 100
memory = []
max_memory = 20000    # Max size of memory
input_dim = 180
output_dim = 2

def reward_function(x):
    if x > 0.8:
        return 100
    return x


def create_q_network(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer="adam", loss='mse')
    return model

q_network = create_q_network(input_dim, output_dim)


def get_action(state, print_qvalue=False):
    if random.uniform(0, 1) < epsilon:
        return random.choices([0,1], [3.7/4, 0.3/4], k=1)[0]  # Random action

    state = np.expand_dims(state, axis=0)
    q_values = q_network.predict(state, verbose = 0)[0]
    if print_qvalue:
        ans = np.argmax(q_values)
        print(f'Predicted: {q_values}', ans)

    return np.argmax(q_values)


def play(count_plays, see_play = False, print_qvalues = False):
    global memory

    env = None
    if see_play:
        env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    else:
        env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=True)
    

    for iteration in range(count_plays):
        
        new_play = []
        frames_last_jump = 100

        env.reset()
        last_obs = np.zeros(input_dim)

        while True:
            
            action = 0
            frames_last_jump += 1

            if frames_last_jump > 10:
                action = get_action(last_obs, print_qvalues)

            if action == 1:
                frames_last_jump = 0

            obs, reward, terminated, _, info = env.step(action)
            reward = reward_function(reward)
            last_obs = obs

            if len(new_play) > 0:
                new_play[-1][3] = last_obs

            new_play.append([last_obs, action, reward, None])

            if terminated:
                break
        
        memory += new_play

    env.close()



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
        print(f'Iteration: {i+1}/{nr_interations}')
        print(f'epsilon: {epsilon}')

        for j in range(nr_trains):
            train_q_network()
        
        if i%10 == nr_interations:
            play(10)

        if len(memory) > max_memory:
            random.shuffle(memory)
            memory = memory[:-(len(memory) - max_memory)]

        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay

        print(f'epsilon: {epsilon}')
        
        if i%10 == 0:
            q_network.save("lidar_model_small.keras")

    q_network.save("lidar_model_small.keras")





#play(20)
#train(100, 50)
#epsilon = 0
#play(5, True, True)

q_network = load_model("lidar_model_small_colab2.keras")
epsilon = 0
play(20, True, True)
