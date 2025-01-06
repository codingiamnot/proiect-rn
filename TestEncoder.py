from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time

autoencoder = load_model('autoencoder_model4.keras')


for i in range(1):
    image = np.load(f"Images\\image{i+1024}.npy", allow_pickle=True)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    image = np.expand_dims(image, axis=0)
    print(image.shape)
    decoded_image = autoencoder.predict(image)
    print(decoded_image.shape)
    plt.imshow(decoded_image[0])
    plt.axis('off')
    plt.show()
    time.sleep(0.08)


print(image.shape)