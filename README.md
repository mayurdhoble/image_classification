***Fruit Classification - A Keras Project with 100% accuracy***

Introduction

This project implements a convolutional neural network (CNN) model using Keras to classify images of three types of fruits: banana, apple, and orange.

Model Architecture

The model employs a sequential architecture with the following layers:

Rescaling (1./255): Normalizes pixel values in the image to the range [0, 1].
Conv2D (16, 3, padding='same', activation='relu'): First convolutional layer with 16 filters of size 3x3, applying ReLU activation for non-linearity. The 'same' padding ensures the output has the same spatial dimensions as the input.
MaxPooling2D(): Downsamples the feature maps to reduce spatial dimensions and computational cost.
Conv2D (32, 3, padding='same', activation='relu'): Second convolutional layer with 32 filters of size 3x3, followed by ReLU activation.
MaxPooling2D(): Another downsampling layer.
Conv2D (64, 3, padding='same', activation='relu'): Third convolutional layer with 64 filters of size 3x3, using ReLU activation.
MaxPooling2D(): More downsampling for feature extraction.
Flatten(): Reshapes the 3D feature maps into a 1D vector for the fully connected layers.
Dropout(0.2): Introduces randomness to prevent overfitting by randomly dropping 20% of neurons during training.
Dense(128): First fully connected layer with 128 neurons.
Dense(len(data_cat)): Output layer with a number of neurons equal to the number of fruit categories (3 in this case). This layer uses a softmax activation to produce class probabilities for each fruit type.
Dataset

This project requires a dataset of images labeled with the corresponding fruit types (banana, apple, orange). Ensure your dataset is properly structured and divided into training, validation, and testing sets.

Usage

1 Install Dependencies: Install the required libraries: tensorflow, keras, and potentially numpy and matplotlib for data manipulation and visualization.

2 Prepare Data: Preprocess your dataset by standardizing image sizes, converting them to NumPy arrays, and applying any necessary normalization. Split your data into training, validation, and testing sets.

3 Compile Model: Compile the model using an appropriate optimizer (e.g., Adam) and a loss function suitable for multi-class classification (e.g., SparseCategoricalCrossentropy). Set metrics to track performance (e.g., accuracy).

4 Train Model: Train the model on your training data, monitoring its performance on the validation set. Adjust hyperparameters (learning rate, number of epochs, etc.) as needed to achieve optimal results.

5 Evaluate Model: Evaluate the trained model on the testing set to assess its generalization ability on unseen data.

6 Predict: Use the trained model to predict the fruit type for new images
