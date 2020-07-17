## Table of Contents

- [Installation](#clone)
- [Features](#features)

---
## Clone

- Clone this repo to your local machine using: `https://github.com/ESGI-MachineLearning-TradeHelper/ModelEngine`

## Installation

Required to get started:
- NumPy
- Pillow
- sklearn
- matplotlib
- Tensorflow
- Keras

## Usage

Run it with Python.

---

## Features

- Linear Regression
    - Classic Linear Regression
- Multi-layer perceptron
    - Trained a model with a lot of Dropout layers to prevent over-fitting.
- Convolutional Neural Network
    - Classic Convolutional Neural Network.
- Residual Neural Network
    - We used a part of <a href="https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py" target="_blank">Keras ResNet50</a> model as example and we fitted it to our need. It appear that they used a different architecture than the orginal paper.
- U-Net
    - We choose a different approche of U-Net. Unlike the orginal paper, our model isn't based on a `Convolutional neural network` but more on a `Resnet` with expansive pathways.
---
