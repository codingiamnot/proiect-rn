from tensorflow.keras.models import *
import numpy as np

autoencoder = load_model('autoencoder_model2.keras')

encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('conv2d_5').output)


def encode_image(image):
    image = np.expand_dims(image, axis=0)
    encoded_image = encoder.predict(image, verbose=0)
    return encoded_image[0].reshape(-1)


image = np.load("Images\\image20.npy", allow_pickle=True)
encode_image(image)
