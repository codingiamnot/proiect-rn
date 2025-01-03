from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np


def load_images():
    nr_images = 229
    ans = []

    for id in range(nr_images):
        ans.append( np.load(f"Images\\image{id+1}.npy", allow_pickle=True))

    return np.array(ans)

def create_autoencoder(input_shape):
    input_img = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Bottleneck (latent space representation)
    encoded = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Assuming RGB image output

    # Autoencoder model
    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


images = load_images()

images = images.astype('float32') / 255.0

X_train, X_val = train_test_split(images, test_size=0.1)

input_shape = (512, 288, 3)
autoencoder = create_autoencoder(input_shape)
autoencoder.summary()

autoencoder.fit(
    X_train, X_train,  # Input and target are the same for autoencoder
    epochs=1,  # Number of epochs
    batch_size=10,  # Batch size
    validation_data=(X_val, X_val),  # Validation data
    shuffle=True)  # Shuffle the data during training

#autoencoder.save_weights('autoencoder_weights.h5')
autoencoder.save('autoencoder_model.keras')

#decoded_image = autoencoder.predict(some_input_image)


