
# MNIST Handwritten Digit Classifier

This project uses machine learning and image processing techniques to recognize handwritten digits from the MNIST dataset. It is implemented using Python with libraries like TensorFlow/Keras, scikit-learn, and OpenCV for training a neural network model.



An implementation of multilayer neural network using keras with an accuracy of 98.314% and using tensorflow with an accuracy over 99%.


## Table of Contents

- Overview
- Dependencies
- Installation
- Usage
- Model Architecture
- Training
- Evaluation
- Results


## Overview
This project is aimed at building a machine learning model that can recognize handwritten digits (0-9). The dataset used is the MNIST dataset, which contains 28x28 grayscale images of handwritten digits. The project involves training a neural network using TensorFlow/Keras to classify these digits.
## About MNIST dataset:
The MNIST database (Modified National Institute of Standards and Technology database) of handwritten digits consists of a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. Additionally, the black and white images from NIST were size-normalized and centered to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.

## About MNIST dataset:
The MNIST database (Modified National Institute of Standards and Technology database) of handwritten digits consists of a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. Additionally, the black and white images from NIST were size-normalized and centered to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.

## Structure of Neural Network:
A neural network is made up by stacking layers of neurons, and is defined by the weights of connections and biases of neurons. Activations are a result dependent on a certain input.

This structure is known as a feedforward architecture because the connections in the network flow forward from the input layer to the output layer without any feedback loops. In this figure:

The input layer contains the predictors.
The hidden layer contains unobservable nodes, or units. The value of each hidden unit is some function of the predictors; the exact form of the function depends in part upon the network type and in part upon user-controllable specifications.
The output layer contains the responses. Since the history of default is a categorical variable with two categories, it is recoded as two indicator variables. Each output unit is some function of the hidden units. Again, the exact form of the function depends in part on the network type and in part on user-controllable specifications.

![Logo](https://camo.githubusercontent.com/fefea3b629321c3dfc0c8c239f1bebc5631b8ac6e8ac52437fbde1bfa3959a1c/687474703a2f2f692e696d6775722e636f6d2f486466656e74422e706e67)


## Summary of Sequential model

![App Screenshot](https://github.com/aakashjhawar/handwritten-digit-recognition/blob/master/assets/model/model_summary.png?raw=true)


## Dependencies
The following libraries are required to run the project:

- Python 3.x
- TensorFlow (>= 2.0)
- Keras
- NumPy
- scikit-learn
- OpenCV
- Matplotlib (for visualizations)
- Pandas (for data handling)
- Jupyter Notebook (optional for experimenting)

You can install these dependencies via pip:

```bash
pip install tensorflow numpy scikit-learn opencv-python matplotlib pandas
```
## Getting Started
How to use :

Install my-project with npm

Clone the repository to your local machine:

```bash
git clone https://github.com/swathi-57/handwritten-digit-recognition.git
cd Handwritten-Digit-Recognition
```
Install all dependencies listed above using pip 
``` bash
pip3 install -r requirements.txt 
python3 tf_cnn.py
```
You can also run the ```bash 
load_model.py ```  to skip the training of NN. It will load the pre saved model from model.json and model.h5 files.

```bash
python3 load_model.py <path/to/image_file>
```
For example
```bash
python3 load_model.py assets/images/1a.jpg 
```
Ensure you have Jupyter installed if you plan to use the notebook format :

```bash
pip install notebook
```
## Usage
#### 1. Load the MNIST dataset:

- The dataset can be loaded directly from TensorFlow/Keras using tensorflow.keras.datasets.mnist.load_data(), which will split the data into training and testing sets.

#### 2. Preprocess the data:

- Normalize pixel values between 0 and 1 for better model performance.
- Reshape data if necessary, e.g., for convolutional neural networks.

#### 3. Build and compile the model:

- The neural network is built using Keras, and typically consists of convolutional layers, dense layers, dropout, and activation functions like ReLU and Softmax.
#### 4. Train the model:

- Use model.fit() with the training data to train the model. You can adjust epochs, batch size, and learning rate to optimize training.
#### 5. Evaluate and predict :

- Use model.evaluate() to check performance on the test dataset.
- Use model.predict() to classify new, unseen images.

#### 6. Run the Jupyter Notebook :

- Open the notebook to explore the code in an interactive environment:
```bash
jupyter notebook
```
## Model Architecture
The model uses a simple Convolutional Neural Network (CNN) with the following layers:

- Conv2D (Convolutional Layer) - Applies convolutional filters to the input image.
- MaxPooling2D - Reduces the spatial dimensions (height and width).
- Flatten - Converts the 2D data into 1D.
- Dense - Fully connected layers.
- Dropout - Prevents overfitting by randomly setting a fraction of input units to zero during training.
- Softmax - For multi-class classification (0-9).
```bash
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
```
## Training
- Compile the model: Use the adam optimizer and sparse_categorical_crossentropy loss function:

``` bash
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
- Fit the model on the training data:

```bash
model.fit(X_train, y_train, epochs=5, batch_size=64)
```
- Monitor performance: Use validation data or early stopping during training to avoid overfitting.

## Evaluation
Evaluate the trained model on the test dataset:

``` bash
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
```




## Result :

- After training, the model can achieve an accuracy of over 98% on the MNIST test set.

  Following image is the prediction of the model.

![Logo](https://github.com/aakashjhawar/Handwritten-Digit-Recognition/blob/master/result.png?raw=true)
