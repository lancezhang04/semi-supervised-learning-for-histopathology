import tensorflow as tf


class BarlowTwinsClassifier(tf.keras.models.Model):
    def __init__(self, encoder, encoder_lr, head_lr, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = encoder
        self.head = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal')
        self.encoder_optimizer = tf.keras.optimizers.SGD(encoder_lr, momentum=0.9)
        self.head_optimizer = tf.keras.optimizers.SGD(head_lr, momentum=0.9)

        inputs = tf.keras.layers.Input((64, 64, 3))
        x = self.encoder(inputs)
        outputs = self.head(x)
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    def summary(self, **kwargs):
        self.model.summary()

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)

    def train_step(self, data):
        X, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self.model(X, training=True)
            loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        encoder_gradients = tape.gradient(loss, self.encoder.trainable_variables)
        head_gradients = tape.gradient(loss, self.head.trainable_variables)

        self.encoder_optimizer.apply_gradients(zip(encoder_gradients, self.encoder.trainable_variables))
        self.head_optimizer.apply_gradients(zip(head_gradients, self.head.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


if __name__ == '__main__':
    import resnet20
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    resnet_enc = resnet20.get_network(
        hidden_dim=1024,
        use_pred=False,
        return_before_head=False,
        input_shape=(64, 64, 3)
    )

    model = BarlowTwinsClassifier(resnet_enc, 0.002, 0.5, 4)
    model.compile(optimizer=tf.keras.optimizers.Adam())
    model.summary()

    datagen = ImageDataGenerator()
    datagen = datagen.flow_from_directory(
        '../../datasets/NuCLS_64_7_grouped_0.2', seed=42, target_size=(64, 64), batch_size=1
    )

    model.fit(datagen, epochs=5)
    # resnet_enc(next(datagen)[0], training=True)
