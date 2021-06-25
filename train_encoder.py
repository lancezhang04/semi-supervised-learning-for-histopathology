from utils.image import image_augmentation
import utils.train.lr_scheduler
from utils.models import resnet20
from utils.train.loss import compute_loss
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, Callback
import logging
import pickle
import os


# I edited the source code of numpy to mitigate a bug/issue
# E:\Northwestern\NuCLS Classification\venv_2\Lib\site-packages\keras_preprocessing\image\iterator.py
# Revert the change later if it causes any issues...

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

# ==================================================================================================================== #
# Configuration
# ==================================================================================================================== #

VERBOSE = 1
BATCH_SIZE = 128
PATIENCE = 10
EPOCHS = 200
IMAGE_SHAPE = [64, 64, 3]
PROJECTOR_DIMENSIONALITY = 2048
ROOT_PATH = 'datasets/NuCLS_64_7'
MODEL_WEIGHTS = 'trained_models/encoder_2048.h5'
SAVE_DIR = 'trained_models'
PREPROCESSING_CONFIG = {
    'color_jittering': 0.8,
    'color_dropping_probability': 0.2,
    'brightness_adjustment_max_intensity': 0.4,
    'contrast_adjustment_max_intensity': 0.4,
    'color_adjustment_max_intensity': 0.2,
    'hue_adjustment_max_intensity': 0.1,
    'gaussian_blurring_probability': [1.0, 0.1],
    'solarization_probability': [0, 0.2]
}

RANDOM_SEED = 42
TRAIN_PATH = os.path.join(ROOT_PATH, 'train')
TEST_PATH = os.path.join(ROOT_PATH, 'test')

# ==================================================================================================================== #
# Load data generators
# P.S. Current implementation is a little hacky
# ==================================================================================================================== #

# Only using training set (and no validation set)
datagen_a = image_augmentation.get_generator(
    PREPROCESSING_CONFIG, view=0
).flow_from_directory(
    TRAIN_PATH,
    seed=RANDOM_SEED,
    target_size=IMAGE_SHAPE[:2],
    batch_size=BATCH_SIZE
)
datagen_b = image_augmentation.get_generator(
    PREPROCESSING_CONFIG, view=1
).flow_from_directory(
    TRAIN_PATH,
    seed=RANDOM_SEED,
    target_size=IMAGE_SHAPE[:2],
    batch_size=BATCH_SIZE
)


def create_dataset(datagen):
    def generator():
        while True:
            # Retrieve the images
            yield datagen.next()[0]
    return tf.data.Dataset.from_generator(generator, output_types='float32')


dataset = tf.data.Dataset.zip((
    create_dataset(datagen_a),
    create_dataset(datagen_b)
))

STEPS_PER_EPOCH = len(datagen_a)
TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS

# ==================================================================================================================== #
# Load model
# ==================================================================================================================== #

# Make sure later that this is the correct model
resnet_enc = resnet20.get_network(
    hidden_dim=PROJECTOR_DIMENSIONALITY,
    use_pred=False,
    return_before_head=False,
    input_shape=IMAGE_SHAPE
)
if MODEL_WEIGHTS:
    resnet_enc.load_weights(MODEL_WEIGHTS)
    if VERBOSE:
        print('Using (pretrained) model weights')

# Load optimizer
WARMUP_EPOCHS = int(EPOCHS * 0.1)
WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)

lr_decay_fn = utils.train.lr_scheduler.WarmUpCosine(
    learning_rate_base=1e-3,
    total_steps=EPOCHS * STEPS_PER_EPOCH,
    warmup_learning_rate=0.0,
    warmup_steps=WARMUP_STEPS
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay_fn)


# Define barlow twins
class BarlowTwins(tf.keras.Model):
    def __init__(self, encoder, lambd=5e-3):
        super(BarlowTwins, self).__init__()
        self.encoder = encoder
        self.lambd = lambd
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

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


# Get model
resnet_enc.trainable = True
barlow_twins = BarlowTwins(resnet_enc)
barlow_twins.compile(optimizer=optimizer)

# ==================================================================================================================== #
# Train model
# ==================================================================================================================== #


# Saves the weights for the encoder only
class ModelCheckpoint(Callback):
    def __init__(self):
        super().__init__()
        self.save_dir = os.path.join(SAVE_DIR, f'encoder_{PROJECTOR_DIMENSIONALITY}.h5')
        self.min_loss = 1e5

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] < self.min_loss:
            self.min_loss = logs['loss']
            print('\nSaving model, new lowest loss:', self.min_loss)
            resnet_enc.save_weights(self.save_dir)


# Might not be the best approach
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=PATIENCE)
mc = ModelCheckpoint()

history = barlow_twins.fit(
    dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[es, mc]
)
