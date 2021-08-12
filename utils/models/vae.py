# Modifed from: https://www.tensorflow.org/tutorials/generative/cvae

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


def build_encoder(image_shape, latent_dim):
    input_ = tf.keras.layers.InputLayer(image_shape)
    layers = [input_]

    for i, filters in enumerate([32, 64, 128, 256, 512]):
        layers.append(downsample_block(filters, name=f'conv_block_{i}'))

    layers.append(tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        # Half is mean, half is logvar
        tf.keras.layers.Dense(latent_dim * 2)
    ]))

    return tf.keras.Sequential(layers, name='projector')


def build_decoder(image_shape, latent_dim):
    input_ = tf.keras.layers.InputLayer(latent_dim)
    layers = [
        input_,
        tf.keras.Sequential([
            # The impact of this Dense layer needs to be investigated further
            tf.keras.layers.Dense(12544),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # For an input shape of (224, 224, 3)
            tf.keras.layers.Reshape((7, 7, 256))
        ], name='projector')
    ]

    for i, filters in enumerate([256, 128, 64, 32, 16]):
        layers.append(upsample_block(filters, name=f'conv_block_{i}'))

    layers.append(tf.keras.layers.Conv2D(filters=image_shape[-1], kernel_size=3,
                                         padding='same', name='conv_output'))

    return tf.keras.Sequential(layers)


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim, image_shape=(128, 128, 3)):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape

        self.encoder = build_encoder(image_shape, latent_dim)
        self.decoder = build_decoder(image_shape, latent_dim)

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

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
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, data, **kwargs):
        # For test time only
        mean, logvar = self.encode(data)
        z = self.reparameterize(mean, logvar)

        return self.decode(z, apply_sigmoid=True)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, x):
        loss = self.compute_loss(x)
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    @staticmethod
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)

        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def get_classifier(config):
    # Pre-trained encoder from a variational autoencoder
    encoder = build_encoder(
        image_shape=config['image_shape'],
        latent_dim=config['latent_dim']
    )
    if config['encoder_weights_path'] is not None:
        print('Loading weights from:', config['encoder_weights_path'])
        encoder.load_weights(config['encoder_weights_path'])

    # Classification head (a single FC layer)
    clf_head = tf.keras.layers.Dense(
        units=config['num_classes'],
        activation='softmax',
        kernel_initializer='he_normal'
    )

    inputs = tf.keras.layers.Input(shape=config['image_shape'])
    x = encoder(inputs)
    outputs = clf_head(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model
