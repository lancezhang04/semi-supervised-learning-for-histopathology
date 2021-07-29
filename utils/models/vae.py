# Based on tutorial: https://www.tensorflow.org/tutorials/generative/cvae

import tensorflow as tf
import numpy as np


def downsample_block(filters, kernel_size=3, name=None):
    block = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size,
                               strides=(2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()
    ], name=name)

    return block


def upsample_block(filters, kernel_size=3, name=None):
    block = tf.keras.Sequential([
        tf.keras.layers.UpSampling2D(interpolation='nearest'),
        tf.keras.layers.Conv2D(filters, kernel_size, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()
    ], name=name)

    return block


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim, image_shape=(128, 128, 3)):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        # Sample vector from distribution
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def build_encoder(self):
        input_ = tf.keras.layers.InputLayer(self.image_shape)
        layers = [input_]

        for i, filters in enumerate([32, 64, 128, 256, 512]):
            layers.append(downsample_block(filters, name=f'conv_block_{i}'))

        layers.append(tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # Half is mean, half is logvar
            tf.keras.layers.Dense(self.latent_dim * 2)
        ]))

        return tf.keras.Sequential(layers, name='projector')

    def build_decoder(self):
        input_ = tf.keras.layers.InputLayer(self.latent_dim)
        layers = [
            input_,
            tf.keras.Sequential([
                tf.keras.layers.Dense(8192),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Reshape((4, 4, 512))
            ], name='projector')
        ]

        for i, filters in enumerate([256, 128, 64, 32, 16]):
            layers.append(upsample_block(filters, name=f'conv_block_{i}'))

        layers.append(tf.keras.layers.Conv2D(filters=self.image_shape[-1], kernel_size=3,
                                             padding='same', name='conv_output'))

        return tf.keras.Sequential(layers)


optimizer = tf.keras.optimizers.Adam(1e-3)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
