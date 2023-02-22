"""
@Created : 08/06/2021
@Edited  : 23/06/2022
@Author  : Alex Poppe
@File    : siamese_cnn_v1.py
@Software: Pycharm
@Description:
This is a Siamese CNN model and (hyper-)parameters for the peak correlation database
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress unnecessary TF loading messages

import tensorflow_addons as tfa
import tensorflow as tf
from utils import distance_metric  # ignore warning. This code is run from another directory


# Define the Siamese CNN model
class sCNN(tf.keras.Model):
    def __init__(self, params):
        super(sCNN, self).__init__()
        # Conv block 1
        self.conv_1a = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation=None,
                                              input_shape=params['s_input_shape'],
                                              kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.instancenorm_1a = tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True,
                                                                beta_initializer='random_uniform',
                                                                gamma_initializer='random_uniform')
        self.lrelu_1a = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.maxpool_1 = tf.keras.layers.MaxPooling2D((2, 2), 2, padding='same')

        # Conv block 2
        self.conv_2a = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.instancenorm_2a = tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True,
                                                                beta_initializer='random_uniform',
                                                                gamma_initializer='random_uniform')
        self.lrelu_2a = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.maxpool_2 = tf.keras.layers.MaxPooling2D((2, 2), 2, padding='same')

        # Conv block 3
        self.conv_3a = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.batchnorm_3a = tf.keras.layers.BatchNormalization()
        self.instancenorm_3a = tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True,
                                                                beta_initializer='random_uniform',
                                                                gamma_initializer='random_uniform')
        self.lrelu_3a = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.maxpool_3 = tf.keras.layers.MaxPooling2D((2, 2), 2, padding='same')

        # Conv block 4
        self.conv_4a = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation=None,
                                              kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.batchnorm_4a = tf.keras.layers.BatchNormalization()
        self.instancenorm_4a = tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True,
                                                                beta_initializer='random_uniform',
                                                                gamma_initializer='random_uniform')
        self.lrelu_4a = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.maxpool_4 = tf.keras.layers.MaxPooling2D((2, 2), 2, padding='same')

        # FC block 1
        self.flattener_5 = tf.keras.layers.Flatten()
        self.dense_5 = tf.keras.layers.Dense(params['s_embedding_dim'], activation=None,
                                             kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.batchnorm_5 = tf.keras.layers.BatchNormalization()
        self.lrelu_5 = tf.keras.layers.LeakyReLU(alpha=0.3)

    def call(self, x, training=False, **kwargs):
        l1 = self.maxpool_1(self.lrelu_1a(self.instancenorm_1a(self.conv_1a(x), training=training)))
        l2 = self.maxpool_2(self.lrelu_2a(self.instancenorm_2a(self.conv_2a(l1), training=training)))
        l3 = self.maxpool_3(self.lrelu_3a(self.instancenorm_3a(self.conv_3a(l2), training=training)))
        l4 = self.maxpool_4(self.lrelu_4a(self.instancenorm_4a(self.conv_4a(l3), training=training)))
        return self.dense_5(self.flattener_5(l4))


# Define the Siamese Dense model/layer
class sDense(tf.keras.Model):
    def __init__(self):
        super(sDense, self).__init__()
        self.lambda_1 = tf.keras.layers.Lambda(distance_metric)
        # (Note: The sigmoid activation is handled by BCE loss function for numerical stability)
        self.dense_1 = tf.keras.layers.Dense(1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.1))

    def call(self, features, **kwargs):
        l1 = self.lambda_1(features)
        return self.dense_1(l1)


# Define the overall Siamese CNN model
class siamese(tf.keras.Model):
    def __init__(self, params):
        super(siamese, self).__init__()

        self.scnn = sCNN(params)
        self.sdense = sDense()

    def call(self, x, training=False, **kwargs):
        x_a, x_b = x  # unpack pair constituents

        embed_a = self.scnn(x_a, training=training)
        embed_b = self.scnn(x_b, training=training)
        out = self.sdense([embed_a, embed_b])

        return out
