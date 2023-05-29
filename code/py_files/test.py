# Test Notebook
## Imports
#Importing basic Libraries/modules. If not installed, you can see to how to install [here](https://github.com/metarex21/Alzheimer-Prediction/tree/main#requirements) in the requirements section.
# Importing TensorFlow and Keras modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten,
    GaussianNoise, GlobalAveragePooling2D, Input, MaxPooling2D, Rescaling,
    Resizing, SeparableConv2D)
import tensorflow.keras.layers as layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Importing data processing and visualization modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Importing scikit-learn modules
from sklearn.metrics import confusion_matrix

# Importing miscellaneous modules
import random
import time
## Loading The Model
model = keras.models.load_model('D:/GitHub/Alzheimer-Prediction/model')
## Constraints

batch_size = 32
img_height = 220
img_width = 220
seed = 42
## Loading Test Data
test_data = tf.keras.utils.image_dataset_from_directory(
    "D:/GitHub/Alzheimer-Prediction/test dataset",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = test_data.class_names
num_classes = len(class_names)

print(f'{num_classes} classes: {class_names}')
AUTOTUNE = tf.data.AUTOTUNEAUTOTUNE = tf.data.AUTOTUNE
test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

# Evaluate the model on the testidation dataset
loss, accuracy = model.evaluate(test_data)

# Calculate the number of misclassified images
num_misclassified = int((1 - accuracy) * len(test_data) * batch_size)

# Print the results
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Number of misclassified images: {num_misclassified} of {len(test_data) * batch_size}")
# Initialize empty lists to store images and labels
test_images = []
test_labels = []

# Iterate through the test dataset and append each batch to a list
for batch in test_data.as_numpy_iterator():
    test_images.append(batch[0])
    test_labels.append(batch[1])

# Concatenate the batches into a single array for both images and labels
test_images = np.concatenate(test_images, axis=0)
test_labels = np.concatenate(test_labels, axis=0)
y_pred = np.array(model.predict(test_images))
y_true = np.array(test_labels)
### Getting most probable label for an image from test dataset
# Use the trained model to predict the labels for the test images
y_pred = model.predict(test_images)

# Convert the predicted probabilities to class labels
y_pred = tf.argmax(y_pred, axis=1).numpy()