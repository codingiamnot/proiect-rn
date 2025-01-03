import constants
import numpy as np
import time

def is_bird(color):
    return (color == constants.BIRD_YELLOW).all()

def is_pipe(color):
    return (color == constants.PIPE_GREEN).all()


def preprocess2(image):
    #a = [[(0,0,0) for j in range(image.shape[1])] for i in range(image.shape[0])]

    start_time = time()

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            #color = (image[row][col][0], image[row][col][1], image[row][col][2])
            if is_bird(image[row][col]):
                #a[row][col] = (255, 255, 255)
                image[row][col] = np.array([255, 255, 255])

def preprocess(image):
    image[np.apply_along_axis(is_bird, 2, image)] = [255, 255, 255]

    return image