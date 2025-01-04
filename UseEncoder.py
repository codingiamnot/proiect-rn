from tensorflow.keras.models import *
import matplotlib.pyplot as plt
import numpy as np

autoencoder = load_model('autoencoder_model.keras')

encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('conv2d_5').output)


def encode_image(image_file):
    image = np.load(image_file, allow_pickle=True)
    image = np.expand_dims(image, axis=0)
    encoded_image = encoder.predict(image)
    return encoded_image[0].reshape(-1)

encode_image("Images\\image20.npy")
