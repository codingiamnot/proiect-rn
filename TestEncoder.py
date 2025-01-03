from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


autoencoder = load_model('autoencoder_model.keras')


image = np.load(f"Images\\image20.npy", allow_pickle=True)
plt.imshow(image)
plt.axis('off')
plt.show()

print(image.shape)
image = np.expand_dims(image, axis=0)
print(image.shape)
decoded_image = autoencoder.predict(image)
plt.imshow(decoded_image[0])
plt.axis('off')
plt.show()

print(image.shape)