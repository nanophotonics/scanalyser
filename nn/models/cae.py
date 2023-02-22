"""
@Created : 13/10/2020
@Edited  : 15/01/2023
@Author  : Alex Poppe
@File    : cae.py
@Software: Pycharm
@Description:
CAE model trained to reconstruct the resting state of BPT SERS spectra
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress unnecessary TF loading messages

import tensorflow_addons as tfa
import tensorflow as tf


# Define the Encoder model
class Encoder(tf.keras.Model):
    def __init__(self, params):
        super(Encoder, self).__init__()
        # Conv block 1
        self.conv_1a = tf.keras.layers.Conv1D(16, 15, padding='same', activation=None,
                                              input_shape=params['c_input_shape'],
                                              kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.groupnorm_1a = tfa.layers.GroupNormalization(groups=1, axis=0)
        self.LReLU_1a = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.maxpool_1a = tf.keras.layers.MaxPooling1D(2, 2, padding='same')

        # Conv block 2
        self.conv_1b = tf.keras.layers.Conv1D(32, 11, padding='same', activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.groupnorm_1b = tfa.layers.GroupNormalization(groups=1, axis=0)
        self.LReLU_1b = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.maxpool_1b = tf.keras.layers.MaxPooling1D(2, 2, padding='same')

        # Conv block 3
        self.conv_1c = tf.keras.layers.Conv1D(64, 7, padding='same', activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.groupnorm_1c = tfa.layers.GroupNormalization(groups=1, axis=0)
        self.LReLU_1c = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.maxpool_1c = tf.keras.layers.MaxPooling1D(2, 2, padding='same')

        # Conv block 4
        self.conv_1d = tf.keras.layers.Conv1D(128, 3, padding='same', activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.groupnorm_1d = tfa.layers.GroupNormalization(groups=1, axis=0)
        self.LReLU_1d = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.maxpool_1d = tf.keras.layers.MaxPooling1D(2, 2, padding='same')

        # FC block 1
        self.flattener = tf.keras.layers.Flatten()
        self.dense_1e = tf.keras.layers.Dense(params['c_embedding_dim'], activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.groupnorm_1e = tfa.layers.GroupNormalization(groups=1, axis=0)
        self.LReLU_1e = tf.keras.layers.LeakyReLU(alpha=0.3)

    def call(self, x, training=False, **kwargs):
        l1 = self.maxpool_1a(self.LReLU_1a(self.groupnorm_1a(self.conv_1a(x))))
        l2 = self.maxpool_1b(self.LReLU_1b(self.groupnorm_1b(self.conv_1b(l1))))
        l3 = self.maxpool_1c(self.LReLU_1c(self.groupnorm_1c(self.conv_1c(l2))))
        l4 = self.maxpool_1d(self.LReLU_1d(self.groupnorm_1d(self.conv_1d(l3))))
        out = self.LReLU_1e(self.groupnorm_1e(self.dense_1e(self.flattener(l4))))
        return out


# Define the Decoder model
class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        # FC block 2
        self.dense_2a = tf.keras.layers.Dense(32 * 128, activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.LReLU_2a = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.reshaper_2a = tf.keras.layers.Reshape((32, 128))

        # Conv block 5
        self.upsample_2b = tf.keras.layers.UpSampling1D(2)
        self.conv_2b = tf.keras.layers.Conv1D(64, 3, padding='same', activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.groupnorm_2b = tfa.layers.GroupNormalization(groups=1, axis=0)
        self.LReLU_2b = tf.keras.layers.LeakyReLU(alpha=0.3)

        # Conv block 6
        self.upsample_2c = tf.keras.layers.UpSampling1D(2)
        self.conv_2c = tf.keras.layers.Conv1D(32, 7, padding='same', activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.groupnorm_2c = tfa.layers.GroupNormalization(groups=1, axis=0)
        self.LReLU_2c = tf.keras.layers.LeakyReLU(alpha=0.3)

        # Conv block 7
        self.upsample_2d = tf.keras.layers.UpSampling1D(2)
        self.conv_2d = tf.keras.layers.Conv1D(16, 11, padding='same', activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.groupnorm_2d = tfa.layers.GroupNormalization(groups=1, axis=0)
        self.LReLU_2d = tf.keras.layers.LeakyReLU(alpha=0.3)

        # Conv block 8 (output)
        self.upsample_2e = tf.keras.layers.UpSampling1D(2)
        self.conv_2e = tf.keras.layers.Conv1D(1, 15, padding='same', activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.1))

    def call(self, x, training=False, **kwargs):
        l1 = self.reshaper_2a(self.LReLU_2a(self.dense_2a(x)))
        l2 = self.LReLU_2b(self.groupnorm_2b(self.conv_2b(self.upsample_2b(l1))))
        l3 = self.LReLU_2c(self.groupnorm_2c(self.conv_2c(self.upsample_2c(l2))))
        l4 = self.LReLU_2d(self.groupnorm_2d(self.conv_2d(self.upsample_2d(l3))))
        out = self.conv_2e(self.upsample_2e(l4))
        return out


# Combine the two models into the complete CAE model
class Autoencoder(tf.keras.Model):
    def __init__(self, params):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder()

    def call(self, x, training=False, **kwargs):
        encoded = self.encoder(x, training=training)
        out = self.decoder(encoded, training=training)
        return out
