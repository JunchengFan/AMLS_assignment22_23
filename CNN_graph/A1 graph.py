# Import the necessary libraries
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt
import keras

# Load your pre-trained model
model = load_model('/Users/fanjuncheng/Desktop/code/A1/save_model/model.h5')

# Choose the layer you want to visualize

fig, axs = plt.subplots(2, 4)
fig.suptitle('Intermediate outputs of layers')

for i in range(8):
    layer_index = i

# Create a new model that outputs the output of the chosen layer
    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[i].output)

# Read the image you want to extract features from
    image = cv2.imread('/Users/fanjuncheng/Desktop/dataset_AMLS_22-23/celeba/img/0.jpg')

    if image is None:
        print("Image not loaded properly")
    else:
        # Resize the image to the same size as the input size of your model
        image = cv2.resize(image, (150, 150))

        # Convert the image to a 4D array
        image = np.expand_dims(image, axis=0)

        # Normalize the image to the range of [-1,1]
        image = keras.applications.mobilenet.preprocess_input(image)

        # Get the output of the chosen layer for the input image
        intermediate_output = intermediate_layer_model.predict(image)

        # Plot the features of each filter in the intermediate layer
        axs[i//4, i%4].imshow(intermediate_output[0,:,:,0], cmap='gray')
        axs[i//4, i%4].set_title("Layer "+str(i))
plt.show()

