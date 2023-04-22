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

#custom loss function for UPAE during training
#gets noise variance and mse. reconstruction loss will be larger 
#in regions with high variance and smaller in regions with low variance

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class Autoencoder:
    def __init__(self, input_shape, multiplier, latentSize, upae=False):
        super(Autoencoder, self).__init__()
        self.upae = upae
        input_layer = Input(shape=input_shape)
        x = Conv2D(int(16*multiplier), 4, strides=2, padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(int(32*multiplier), 4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(int(64*multiplier), 4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(int(64*multiplier), 4, strides=2, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        volumeSize = K.int_shape(x)

        #Latent representation Encoder
        
        if upae is True:
            print("UPAE")
            latent_enc = Flatten()(x)
            latent_enc = Dense(2048, activation='relu')(latent_enc)
            latent_enc = Dense(latentSize)(latent_enc)
            z_mean = Dense(latentSize)(latent_enc)
            z_log_var = Dense(latentSize)(latent_enc)
            latentInputs = Input(shape=(latentSize,))
            latentOutputs = Lambda(sampling, output_shape=(latentSize,))([z_mean, z_log_var])
            self.z_mean = z_mean
            self.z_log_var = z_log_var

        else:
            print("Vanilla AE")
            latent_enc = Flatten()(x)
            latent_enc = Dense(2048, activation='relu')(latent_enc)
            latent_enc = Dense(latentSize*2)(latent_enc)
            latentInputs = Input(shape=(latentSize*2,))

        self.encoder = Model(input_layer, latent_enc, name="encoder")

        
        #Latent representation Decoder
        print("Decoder")
        latent_dec = Dense(2048, activation='relu')(latentInputs)
        latent_dec = Dense(int(64 * multiplier) * volumeSize[1]*volumeSize[2])(latent_dec)
        latent_dec = Reshape((volumeSize[1], volumeSize[2], int(64*multiplier)))(latent_dec)
        latent_dec = BatchNormalization()(latent_dec)


        x = Conv2DTranspose(int(64*multiplier), 4, strides=2, padding='same')(latent_dec)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(int(32*multiplier), 4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(int(16*multiplier), 4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(3, 4, strides=2, padding='same')(x)
        outputs = Activation("relu")(x)
                    
        self.decoder = Model(latentInputs, outputs, name="decoder")

        self.autoencoder = Model(input_layer, self.decoder(self.encoder(input_layer)),
            name="autoencoder")

    
    def compile_AE(self,upae=False):
        #learning rate similar to Mao et al's
        optimizer = Adam(learning_rate=0.0005)

        #custom loss function for UPAE with noise variance 
        def custom_loss(y_true, y_pred):
            rec_err = tf.math.squared_difference(self.z_mean, y_true) #mse
            loss1 = tf.math.reduce_mean(tf.math.exp(-self.z_log_var) * rec_err) 
            loss2 = tf.math.reduce_mean(self.z_log_var)
            loss = loss1 + loss2
            return loss
        
        if upae is True:
            self.autoencoder.compile(optimizer = optimizer, 
                                 loss=custom_loss, 
                                 metrics= ['mse', 'accuracy',
                                           AUC(name="AUC"),
                                           Precision(name="Precision"),
                                           Recall(name='Recall'),
                                           TruePositives(name="True Positives"),
                                           FalsePositives(name="False Positives")])
            
        #MSE loss function only for vanilla AE
        else:
            self.autoencoder.compile(optimizer = optimizer, 
                                 loss='mse', 
                                 metrics= ['accuracy',
                                           AUC(name="AUC"),
                                           Precision(name="Precision"),
                                           Recall(name='Recall'),
                                           TruePositives(name="True Positives"),
                                           FalsePositives(name="False Positives")])
        return
        


    def fit_AE(self, x_train, y_train, x_test, epochs = 1, batch_size = 32 ):
        log_dir = "logs/fit/"
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        return self.autoencoder.fit(x_train, 
                                    y_train, 
                                    epochs = epochs, 
                                    batch_size=batch_size, 
                                    validation_data=(x_test, x_test), 
                                    callbacks=[tensorboard_callback])

    
    #evaluate model performance using validation set
    def validate(self, x_test, batch_size, upae=False):
        if upae is True:
            score = self.autoencoder.evaluate(x_test, x_test, batch_size)
            print('MSE + Noise Variance', score[0])
            print('Accuracy', score[1])
            print('AUC', score[2])
            print('Precision', score[3])
        else:
            score = self.autoencoder.evaluate(x_test, x_test, batch_size)
            print('MSE', score[0])
            print('Accuracy', score[1])
            print('AUC', score[2])
            print('Precision', score[3])

        return 
    

            
