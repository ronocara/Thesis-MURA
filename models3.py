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
from tensorflow import keras
import torch
import numpy as np


import matplotlib.pyplot as plt
import numpy as np


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        #the random vector taken from mean and log var
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim)) 

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    

class VAE(keras.Model):
    def __init__(self, encoder, decoder, upae=False, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    #will run during fit()
    def train_step(self, data):
        with tf.GradientTape() as tape:
                print("Vanilla Loss")
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z) #reconstructed image
                mse_loss = tf.reduce_mean(tf.square(tf.cast(data, tf.float32) - tf.cast(reconstruction, tf.float32)))
                total_loss = mse_loss

            
            #     print("UPAE Loss")
            #     z_mean, z_log_var, z = self.encoder(data)
            #     reconstruction = self.decoder(z)
            #     reconstruction_loss = tf.reduce_mean(
            #         tf.reduce_sum(
            #             keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            #         )
            #     )
            #     kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            #     kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            #     total_loss = reconstruction_loss + kl_loss

        #calculate gradients using back propagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        #updating the metrics trackers 
        self.total_loss_tracker.update_state(total_loss)
        # self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        # self.kl_loss_tracker.update_state(kl_loss)

        return {
            "mse_loss": self.total_loss_tracker.result(),
            # "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            # "kl_loss": self.kl_loss_tracker.result(),
        }

class UPAE(keras.Model):
    def __init__(self, encoder, decoder, upae=False, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    #will run during fit()
    def train_step(self, data):
        with tf.GradientTape() as tape:
            
                print("UPAE Loss")
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                    )
                )
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = reconstruction_loss + kl_loss

        #calculate gradients using back propagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        #updating the metrics trackers 
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "mse_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }



        
