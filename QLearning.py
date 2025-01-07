from tensorflow.keras.models import Sequential
from tensorflow.keras.models import *
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
import gymnasium as gym
import flappy_bird_gymnasium
import random

from UseEncoder import *
import pygame
import time
import keyboard

gamma = 0.9         # Discount factor
epsilon = 0.8        # Initial exploration rate
epsilon_min = 0.1    # Minimum exploration rate
epsilon_decay = 0.995  # Decay factor for exploration
learning_rate = 0.1
batch_size = 8
memory = []
max_memory = 2000    # Max size of memory
input_dim = 576#288
output_dim = 2
last_nr_play = 0

def create_q_network(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(16, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    global learning_rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

q_network = create_q_network(input_dim, output_dim)

def load_memory(nrPlays):
    for id in range(nrPlays):
        file_name = f"HumanPlay2/play{id}.txt"
        file = open(file_name, "r")
        words = file.read()
        words = words.split(' ')
        nxt_word = 0

        nr_moments = int(words[nxt_word])

        nxt_word += 1
        n_play = []

        for moment_id in range(nr_moments):

            image = []
            for id in range(input_dim):
                x = float(words[nxt_word])
                nxt_word += 1

                image.append(x)

            image = np.array(image)
            image = np.expand_dims(image, axis=0)

            action = int(words[nxt_word])
            nxt_word += 1

            reward = float(words[nxt_word])
            nxt_word += 1

            n_play.append((image, action, reward))

        memory.append(n_play)

        file.close()


def get_action(state, print_qvalue):
    if random.uniform(0, 1) < epsilon:
        return random.choices([0,1], [10/20, 10/20], k=1)[0]  # Random action

    q_values = q_network.predict(state, verbose = 0)
    if print_qvalue:
        print(f'Predicted: {q_values}')
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
                reward += 10
            elif terminated:
                reward -= 100

            new_play.append((encoded_image, action, reward))

            if terminated:
                memory.append(new_play)
                break

    env.close()

    nr = last_nr_play
    for play in memory:
        file_name = f"HumanPlay2/play{nr}.txt"
        file = open(file_name, "w")

        file.write(f"{len(play)} ")
        for moment in play:

            for x in moment[0][0]:
                file.write(f"{x} ")

            file.write(f"{moment[1]} ")
            file.write(f"{moment[2]} ")

        nr += 1
        file.close()



def play(count_plays, see_play = False, print_qvalues = False):
    count_frame = 0
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

            count_frame = count_frame + 1
            action = 0
            if count_frame == 5:
                count_frame = 0
                action = get_action(encoded_image, print_qvalues)

            if see_play:
                surface = pygame.surfarray.make_surface(image.transpose(1, 0, 2))
                screen.blit(surface, (0, 0))
                pygame.display.flip()

            obs, reward, terminated, _, info = env.step(action)

            new_play.append((encoded_image, action, reward))

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
                states_q.append(batch[play_id][state_id][0][0])

            states_q = np.array(states_q)
            q_values = q_network.predict(states_q, verbose=0)

            for state_id in range(len(batch[play_id])):

                state, action, reward = batch[play_id][state_id]

                if state_id == len(batch[play_id]) - 1:
                    states.append(state[0])
                    targets.append(np.array([reward, reward]))
                else:
                    target = reward + gamma * np.max(q_values[state_id + 1])
                    target_f = q_values[state_id]
                    target_f[action] = target
                    states.append(state[0])
                    targets.append(target_f)

        print(iteration)
        q_network.fit(np.array(states), np.array(targets), epochs=4, verbose=1)

def train(nr_interations, epochs):
    global epsilon
    global memory

    memory.clear()
    #load_memory(5)

    for i in range(nr_interations):
        print(f'Iteration: {i+1}/{nr_interations}')

        train_q_network(epochs)
        play(10)

        if len(memory) > max_memory:
            random.shuffle(memory)
            memory = memory[:-10]

        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay

    print(f'epsilon: {epsilon}')

#play_by_hand(16)
train(64,1)
q_network.save('q_network_model_test.keras')

#q_network = load_model('q_network_model_test.keras')

#epsilon = 0
#play(5, True, True)


