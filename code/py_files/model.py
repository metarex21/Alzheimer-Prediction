# Importing TensorFlow and Keras modules
import tensorflow as tf
from tensorflow import keras
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

#GPU

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
len(gpus)

## Training constansts
batch_size = 32
img_height = 220
img_width = 220
seed = 69

## Load data
# Data is being Split here into 80:20 ratio where 80% is being used for training; rest for validation. You can find the dataset [here](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset).
### 1.Training Dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    "D:/GitHub/Alzheimer-Prediction/dataset",
    subset='training',
    validation_split=0.2,
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

### 2.Validation Dataset
val_data = tf.keras.utils.image_dataset_from_directory(
    "D:/GitHub/Alzheimer-Prediction/dataset",
    subset='validation',
    validation_split=0.2,
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

## Classes Names
# Returns the number of classes and the corresponding class names in the train_data dataset by printing in output
class_names = train_data.class_names
num_classes = len(class_names)

print(f'{num_classes} classes: {class_names}')
## Preparing dataset for training and evaluation
AUTOTUNE = tf.data.AUTOTUNEAUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)
## Plotting example images
def plot_images(n_rows=3, n_cols=7, data=train_data):
    plt.figure(figsize=(n_cols*2, int(n_rows*1.8)))
    for images, labels in data.take(int((n_rows * n_cols) / batch_size) + 1): # "take" takes random batch 
        for i in range(n_rows*n_cols):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(images[i].numpy().astype("uint16"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
plot_images()
## Getting image and label batch shape
for image_batch, labels_batch in train_data:
    image_shape = image_batch.shape
    print(f"Image batch shape: {image_shape}")
    print(f"Label batch shape: {labels_batch.shape}")
    break
## Model
### 1. Creating normalization layer
normalization_layer = tf.keras.layers.Rescaling(scale=1./255)
### Branch
input_layer = Input(shape=(img_width, img_height, 3))
gaussian_noise = GaussianNoise(0.15)(input_layer)
branch_1 = normalization_layer(gaussian_noise)

branch_1 = Conv2D(32, (3,3), padding='same', activation="relu")(branch_1)
branch_1 = BatchNormalization()(branch_1)
branch_1 = MaxPooling2D((2, 2))(branch_1)

branch_1 = Conv2D(64, (3,3), padding='same', activation="relu")(branch_1)
branch_1 = BatchNormalization()(branch_1)
branch_1 = MaxPooling2D((2, 2))(branch_1)

branch_1 = Conv2D(128, (3,3), padding='same', activation="relu")(branch_1)
branch_1 = BatchNormalization()(branch_1)
branch_1 = MaxPooling2D((2, 2))(branch_1)

branch_1 = Conv2D(64, (3,3), padding='same', activation="relu")(branch_1)
branch_1 = BatchNormalization()(branch_1)
branch_1 = MaxPooling2D((2, 2))(branch_1)

branch_1 = Flatten()(branch_1)
output_layers = Dense(256, activation='relu')(branch_1)
output_layers = Dense(128, activation='relu')(output_layers)

output_layers = Dropout(0.5)(output_layers)

output_layers = Dense(num_classes, activation='softmax')(output_layers)
### Creating model instance
model = tf.keras.Model(input_layer, output_layers)
### Compiling an optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
### Model description
tf.keras.utils.plot_model(model, show_shapes=True, dpi=150)
model.summary()
## Callbacks
### 1. Early stopping
early_stopping= EarlyStopping(monitor='val_loss', patience=4,start_from_epoch=3,restore_best_weights=True)
### 2. ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-7)
## Training
epochs = 40

history = model.fit(
  train_data,
  epochs = epochs,
  validation_data = val_data,
  batch_size = batch_size,
  callbacks = [early_stopping, reduce_lr],
  verbose = 1
)
## Model training and validation metrics
# Extract metrics from history object
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Plot accuracy metrics
ax1.plot(acc, label='Training Accuracy', color='blue', linestyle='solid')
ax1.plot(val_acc, label='Validation Accuracy', color='orange', linestyle='dashed')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Training and Validation Accuracy', fontsize=10)
ax1.grid()

# Plot loss metrics
ax2.plot(loss, label='Training Loss', color='blue', linestyle='solid')
ax2.plot(val_loss, label='Validation Loss', color='orange', linestyle='dashed')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Training and Validation Loss', fontsize=10)
ax2.grid()

# Add legend and title to figure
fig.legend(loc='lower center', ncol=2, fontsize=10)
fig.suptitle('Model Training History', fontsize=16)

# Show the plot
plt.show()
## Model evaluation on validation dataset
# Evaluate the model on the validation dataset
loss, accuracy = model.evaluate(val_data)

# Calculate the number of misclassified images
num_misclassified = int((1 - accuracy) * len(val_data) * batch_size)

# Print the results
print(f"Validation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")
print(f"Number of misclassified images: {num_misclassified} of {len(val_data) * batch_size}")
plot_images(data=val_data)
## Unpacking batches to create confusion matrix
# Initialize empty lists to store images and labels
val_images = []
val_labels = []

# Iterate through the test dataset and append each batch to a list
for batch in val_data.as_numpy_iterator():
    val_images.append(batch[0])
    val_labels.append(batch[1])

# Concatenate the batches into a single array for both images and labels
val_images = np.concatenate(val_images, axis=0)
val_labels = np.concatenate(val_labels, axis=0)
y_pred = np.array(model.predict(val_images))
y_true = np.array(val_labels)
### Getting most probable label for an image from test dataset
# Use the trained model to predict the labels for the test images
y_pred = model.predict(val_images)

# Convert the predicted probabilities to class labels
y_pred = tf.argmax(y_pred, axis=1).numpy()
### Creating a list with missclassified labels
# Get the indices of the incorrectly classified images
incorrect_indices = np.nonzero(y_pred != y_true)[0]

# Get the incorrectly classified images and labels
false_class = list(zip(val_images[incorrect_indices], y_pred[incorrect_indices], y_true[incorrect_indices]))

# Get the correctly classified images and labels
correct_indices = np.nonzero(y_pred == y_true)[0]
true_class = list(zip(val_images[correct_indices], y_pred[correct_indices], y_true[correct_indices]))
## Confusion Matrix
# Compute the confusion matrix for the predicted labels and true labels
cm = confusion_matrix(y_true, y_pred, normalize='true')

# Create a heatmap visualization of the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# Set tick labels
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

# Rotate the tick labels and set axis labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlabel('Predicted')
ax.set_ylabel('True')

# Loop over data to create annotations
for i in range(len(class_names)):
    for j in range(len(class_names)):
        text = ax.text(j, i, format(cm[i, j], '.2f'),
                       ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")

# Set title and show the plot
ax.set_title("Confusion matrix of the classifier")
fig.tight_layout()
plt.show()
## Plotting errors
def plot_predictions(labels, cols=5):
    number_of_misscl = round((1 - accuracy) * len(val_data) * batch_size)
    random.shuffle(labels)
    rows = int(number_of_misscl/cols)-1
    fig = plt.figure(figsize=(cols*10, 45))
    for i in range(1, cols*rows +1):
        fig.add_subplot(rows, cols, i)
        plt.title(f'Pred: {class_names[labels[i][1]]}\nTrue: {class_names[labels[i][2]]}',fontsize=40)
        plt.axis("off")
        plt.imshow(labels[i][0].astype("uint16"))
    plt.tight_layout()
plot_predictions(false_class, cols=5)
## Saving The Model
model.save("D:/GitHub/Alzheimer-Prediction/model")