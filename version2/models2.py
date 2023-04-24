import tensorflow.keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, BatchNormalization, Activation
from tensorflow.keras.metrics import AUC, Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.losses as losses
import tensorflow as tf
import torch

import matplotlib.pyplot as plt
import numpy as np


class Autoencoder:
    def __init__(self, input_shape, multiplier, latentSize, upae=False):
        super(Autoencoder, self).__init__()

        input_layer = Input(shape=input_shape)
    
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(int(16*multiplier), 4, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv2D(int(32*multiplier), 4, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv2D(int(64*multiplier), 4, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv2D(int(64*multiplier), 4, strides=2, padding='same'),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.BatchNormalization(),
        ])
        if upae is True:
            print("UPAE")  
            self.linear_enc = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2048, activation='relu'),
                tf.keras.layers.Dense(latentSize)
            ]) 
        else:
            print("Vanilla AE")
            self.linear_enc = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2048, activation='relu'),
                tf.keras.layers.Dense(latentSize*2)
            ]) 
        self.linear_dec = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(latentSize*2)
        ]) 
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(int(64*multiplier), 4, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv2DTranspose(int(32*multiplier), 4, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv2DTranspose(int(16*multiplier), 4, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same'),
            tf.keras.layers.Activation('relu')
        ])
    

    
    def encoder(self, x):
        lat_rep = self.encoder(x)
        lat_rep = self.linear_enc(lat_rep)
        return lat_rep
        
    def decoder(self, x):
        out = self.linear_dec(x)
        out = self.decoder(out)
        return out
        
    def forward(self, x):
        lat_rep = self.encoder(x)
        out = self.decoder(lat_rep)
        return out
        
