# ------------------------------------------------------------- #
# ------------------------- IMPORTING ------------------------- #
# ------------------------------------------------------------- #

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
#from ResNet_Weights import ResNet18
from keras.models import Sequential , Model
from keras import layers
from keras.layers.core import Dense, Dropout 
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import SGD , Adam
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor
from keras.applications import MobileNetV2, inception_v3
from keras.applications.densenet import *
from keras.applications.xception import *
# import efficientnet.keras as efn
from keras import backend as K   # session cleaning for gridsearch

# ------------------------------------------------------------- #
# ------------------------- MobileNet ------------------------- #
# ------------------------------------------------------------- #

def create_MobileNetV2(learn_rate=0.01, num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD' ):
    K.clear_session()
    # configure model and training parameters
    image_size = 224
    if (optimizer=="SGD"):
        optimizer = SGD(lr=learn_rate)
    elif (optimizer=="Adam"):
        optimizer = Adam(lr=learn_rate)

    # Load the MobileNet model and initialize to imagenet
    mob_net = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    model = Sequential()
    model.add(mob_net)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(dense_neurons, activation='relu'))
    if (num_classes==2):
        model.add(Dense(1, activation=activation))
    else:
        model.add(Dense(num_classes, activation=activation))
    # unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    model.summary()

    if (num_classes==2):
        model.compile(loss="binary_crossentropy", optimizer=optimizer,
                    metrics=["accuracy"])
    else:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                    metrics=["accuracy"])

    return model

# ------------------------------------------------------------- #
# ------------------------- Inception ------------------------- #
# ------------------------------------------------------------- #
def create_InceptionV3(learn_rate=0.01,num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'):
    K.clear_session()

    # configure model and training parameters
    image_size = 299
    if (optimizer=="SGD"):
        optimizer = SGD(lr=learn_rate)
    elif (optimizer=="Adam"):
        optimizer = Adam(lr=learn_rate)

    # Getting the InceptionV3 model 
    inceptionV3 = InceptionV3(weights = 'imagenet', include_top = False, input_shape=(image_size, image_size, 3))
    model = Sequential()
    model.add(inceptionV3)
    #   model.add(layers.Flatten())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(dense_neurons, activation='relu'))
    if (num_classes==2):
        model.add(Dense(1, activation=activation))
    else:
        model.add(Dense(num_classes, activation=activation))
    # first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers:
        layer.trainable = True

    from contextlib import redirect_stdout
    # with open('modelsummary.txt', 'w') as f:
    #     with redirect_stdout(f):
    #         model.summary()
    model.summary()

    # initialize our initial learning rate and # of epochs to train for


    if (num_classes==2):
        model.compile(loss="binary_crossentropy", optimizer=optimizer,
                    metrics=["accuracy"])
    else:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                    metrics=["accuracy"])


    return model


# ------------------------------------------------------------- #
# ------------------------- Xception ------------------------- #
# ------------------------------------------------------------- #
def create_Xception(learn_rate=0.01,num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'):
    K.clear_session()
    # configure model and training parameters
    image_size = 299
    if (optimizer=="SGD"):
        optimizer = SGD(lr=learn_rate)
    elif (optimizer=="Adam"):
        optimizer = Adam(lr=learn_rate)

    # Load the Xceptin model with 3 classes and initialized to imagenet
    xcep = Xception(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    model = Sequential()
    model.add(xcep)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(dense_neurons, activation='relu'))
    if (num_classes==2):
        model.add(Dense(1, activation=activation))
    else:
        model.add(Dense(num_classes, activation=activation))
    # unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    model.summary()

    if (num_classes==2):
        model.compile(loss="binary_crossentropy", optimizer=optimizer,
                    metrics=["accuracy"])
    else:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                    metrics=["accuracy"])

    return model

# ------------------------------------------------------------- #
# --------------------- EfficientNetB1 ------------------------ #
# ------------------------------------------------------------- #

def create_EfficientNetB1(learn_rate=0.01,num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'):
    K.clear_session()
    # configure model and training parameters
    image_size = 224
    if (optimizer=="SGD"):
        optimizer = SGD(lr=learn_rate)
    elif (optimizer=="Adam"):
        optimizer = Adam(lr=learn_rate)

    base_model = efn.EfficientNetB1(weights = 'imagenet', include_top = False, input_shape=(image_size, image_size, 3))

    # Adding a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    #Adding a fully-connected dense layer
    x = Dense(dense_neurons, activation='relu')(x)

    #Adding a logistic layer - We have 6 classes 
    if (num_classes==2):
        predictions = Dense(1, activation=activation)(x)
    else:
        predictions = Dense(num_classes, activation=activation)(x)

    # The model we will train
    model = Model(inputs = base_model.input, outputs = predictions)

    # first: train only the top layers i.e. freeze all convolutional 
    for layer in base_model.layers:
        layer.trainable = True

    if (num_classes==2):
        model.compile(loss="binary_crossentropy", optimizer=optimizer,
                    metrics=["accuracy"])
    else:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                    metrics=["accuracy"])

    return model


# ------------------------------------------------------------- #
# ------------------------- ResNet18 -------------------------- #
# ------------------------------------------------------------- #

# def create_ResNet18(learn_rate=0.01,num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'):
#     K.clear_session()
    
#     image_size = 224
#     if (optimizer=="SGD"):
#         optimizer = SGD(lr=learn_rate)
#     elif (optimizer=="Adam"):
#         optimizer = Adam(lr=learn_rate)

#     # Load the MobileNet model with 3 classes and initialized to imagenet
#     resnet18 = ResNet18(input_shape=(image_size, image_size, 3), weights='imagenet', classes=3, include_top=False)
#     model = Sequential()
#     model.add(resnet18)
#     model.add(GlobalAveragePooling2D())
#     model.add(Dense(dense_neurons, activation='relu'))
#     if (num_classes==2):
#         model.add(Dense(1, activation=activation))
#     else:
#         model.add(Dense(num_classes, activation=activation))

#     # unfreeze all layers
#     for layer in model.layers:
#         layer.trainable = True

#     model.summary()

#     if (num_classes==2):
#         model.compile(loss="binary_crossentropy", optimizer=optimizer,
#                     metrics=["accuracy"])
#     else:
#         model.compile(loss="categorical_crossentropy", optimizer=optimizer,
#                     metrics=["accuracy"])

#     return model


# ------------------------------------------------------------- #
# ----------------------- DenseNet121 ------------------------- #
# ------------------------------------------------------------- #
# create densenet model function
def create_densenet121(learn_rate=0.01,num_classes=2, dense_neurons=512,activation = 'softmax', optimizer='SGD'):
    K.clear_session()
    # configure model and training parameters
    image_size = 224
    if (optimizer=="SGD"):
        optimizer = SGD(lr=learn_rate)
    elif (optimizer=="Adam"):
        optimizer = Adam(lr=learn_rate)
   
    # Load the densenet model initialized to imagenet
    denseN =  DenseNet121(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    model = Sequential()
    model.add(denseN)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(dense_neurons, activation='relu'))
    if (num_classes==2):
        model.add(Dense(1, activation=activation))
    else:
        model.add(Dense(num_classes, activation=activation))
    # unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    model.summary()

    if (num_classes==2):
        model.compile(loss="binary_crossentropy", optimizer=optimizer,
                    metrics=["accuracy"])
    else:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                    metrics=["accuracy"])

    return model
