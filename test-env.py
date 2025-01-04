import flappy_bird_gymnasium
import gymnasium as gym
import keyboard
from tensorflow.python.autograph.core.converter_testing import allowlist

from Preprocessing import preprocess
import pygame
import time
import pandas as pd
import numpy as np

screen = pygame.display.set_mode((288, 512))
nrSavedImages = 138
lastSaved = -100
currentImage = -1

def save(image):
    global nrSavedImages
    nrSavedImages += 1

    file_name = f"image{nrSavedImages}"

    np.save(f"Images/{file_name}", image, allow_pickle=True)

def play_flappy_bird():
    global currentImage
    global lastSaved

    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    obs, _ = env.reset()

    while True:
        action = 0
        if keyboard.is_pressed('space'):
            action = 1
        if keyboard.is_pressed('esc'):
            print("Exiting the game.")
            break

        image = env.render()

        currentImage += 1

        if currentImage - lastSaved > 5:
            save(image)
            lastSaved = currentImage

        surface = pygame.surfarray.make_surface(image.transpose(1,0,2))

        screen.blit(surface, (0, 0))
        time.sleep(0.08)
        pygame.display.flip()

        obs, reward, terminated, _, info = env.step(action)

        if reward > 0:
            reward += 10  #bonus for passing pipes
        elif terminated:  #the bird hits the ground or a pipe
            reward -= 20  #penalize heavily for losing

        if terminated:
            print("Game Over!")
            obs, _ = env.reset()

    env.close()

play_flappy_bird()
