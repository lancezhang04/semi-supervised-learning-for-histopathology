from utils.train.loss import compute_loss
import tensorflow as tf


class BarlowTwins(tf.keras.Model):
    def __init__(self, encoder, blur_layer, preprocessing_config, lambd=5e-3):
        super(BarlowTwins, self).__init__()
        self.encoder = encoder
        self.blur_layer = blur_layer
        self.blur_probabilities = preprocessing_config['gaussian_blurring_probability']
        self.lambd = lambd
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        # Gaussian blurring
        ds_one = tf.cond(
            tf.random.uniform(shape=[]) < self.blur_probabilities[0],
            true_fn=lambda: self.blur_layer(ds_one),
            false_fn=lambda: ds_one
        )
        ds_two = tf.cond(
            tf.random.uniform(shape=[]) < self.blur_probabilities[0],
            true_fn=lambda: self.blur_layer(ds_two),
            false_fn=lambda: ds_two
        )

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z_a, z_b = self.encoder(ds_one, training=True), self.encoder(ds_two, training=True)
            loss = compute_loss(z_a, z_b, self.lambd)

        # Compute gradients and update the parameters.
        gradients = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
