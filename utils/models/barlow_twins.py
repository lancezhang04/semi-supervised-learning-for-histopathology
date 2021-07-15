from utils.train.loss import compute_loss
import tensorflow as tf


class BarlowTwins(tf.keras.Model):
    def __init__(self, encoder, blur_layer, preprocessing_config, batch_size, lambd=5e-3):
        super(BarlowTwins, self).__init__()
        self.encoder = encoder
        self.blur_layer = blur_layer

        self.blur_probabilities = preprocessing_config['gaussian_blurring_probability']
        self.batch_size = batch_size
        self.lambd = lambd
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        return [self.loss_tracker]

    def blur_images(self, imgs, prob):
        # Randomly blurs a batch of images according to a given probability

        probs = tf.random.uniform(shape=[self.batch_size])  # [imgs.shape[0]])
        change_indices = tf.cast(tf.where(probs <= prob), 'int32')
        keep_indices = tf.cast(tf.where(probs > prob), 'int32')
        # print(len(change_indices))

        shape = tf.shape(imgs)

        imgs_blurred = self.blur_layer(imgs)
        imgs_blurred = tf.gather(imgs_blurred, change_indices[:, -1])
        imgs_keep = tf.gather(imgs, keep_indices[:, -1])

        imgs = tf.scatter_nd(change_indices, imgs_blurred, shape)
        imgs = tf.tensor_scatter_nd_add(imgs, keep_indices, imgs_keep)

        return imgs

    def call(self, data, **kwargs):
        return self.encoder(data[0], training=True), self.encoder(data[1], training=True)

    def train_step(self, data):
        # Unpack the data
        imgs_a, imgs_b = data

        # Gaussian blurring (faster on GPU)
        imgs_a = self.blur_images(imgs_a, self.blur_probabilities[0])
        imgs_b = self.blur_images(imgs_b, self.blur_probabilities[1])

        # Forward pass through the encoder and predictor
        with tf.GradientTape() as tape:
            z_a, z_b = self.encoder(imgs_a, training=True), self.encoder(imgs_b, training=True)
            loss = compute_loss(z_a, z_b, self.lambd)

        # Compute gradients and update the parameters
        gradients = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
