from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from utils import image_augmentation
from utils.train import lr_scheduler
from utils.models import resnet20
from utils.models.barlow_twins import BarlowTwins
from utils.datasets import get_dataset_df
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import pickle
import os

import numpy as np
from tqdm import tqdm


# ==================================================================================================================== #
# Configuration
# ==================================================================================================================== #
# region

VERBOSE = 1
BATCH_SIZE = 32
PATIENCE = 30
EPOCHS = 1
IMAGE_SHAPE = [32, 32, 3]
PROJECTOR_DIMENSIONALITY = 2048

PREPROCESSING_CONFIG = {
    'vertical_flip_probability': 0.5,
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

MODEL_WEIGHTS = None  # 'trained_models/encoder_2048.h5'
SAVE_DIR = ''  # 'trained_models'

DATASET_CONFIG = {
    'split': 'tissue_classification/fold_test.csv',
    'train_split': 1,
    'validation_split': 0,
    'dataset_dir': 'tissue_classification/tissue_classification',
    'groups': {},
    'major_groups': []
}
# endregion


# ==================================================================================================================== #
# Load data generators
# P.S. Current implementation is a little hacky
# ==================================================================================================================== #
# region

# Only using training set (and no validation set)
df = get_dataset_df(DATASET_CONFIG, RANDOM_SEED)

aug_a = image_augmentation.get_preprocessing_function(PREPROCESSING_CONFIG, view=0)
datagen_a = ImageDataGenerator(
    preprocessing_function=lambda x: aug_a(image=x)
).flow_from_dataframe(
df[df['split'] == 'train'],
    seed=RANDOM_SEED,
    target_size=IMAGE_SHAPE[:2], batch_size=BATCH_SIZE
)

aug_b = image_augmentation.get_preprocessing_function(PREPROCESSING_CONFIG, view=1)
datagen_b = ImageDataGenerator(
    preprocessing_function=lambda x: aug_b(image=x)
).flow_from_dataframe(
df[df['split'] == 'train'],
    seed=RANDOM_SEED,
    target_size=IMAGE_SHAPE[:2], batch_size=BATCH_SIZE
)

STEPS_PER_EPOCH = len(datagen_a)
TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS

X_a, X_b = next(datagen_a)[0], next(datagen_b)[0]
for i in tqdm(range(STEPS_PER_EPOCH - 1), ncols=100):
    X_a = np.concatenate([X_a, next(datagen_a)[0]], axis=0)
    X_b = np.concatenate([X_b, next(datagen_b)[0]], axis=0)
X_a = X_a.astype('float32') / 127.5 - 1
X_b = X_b.astype('float32') / 127.5 - 1

dataset_a = tf.data.Dataset.from_tensor_slices(X_a)
dataset_b = tf.data.Dataset.from_tensor_slices(X_b)

dataset_a = dataset_a.batch(BATCH_SIZE)
dataset_b = dataset_b.batch(BATCH_SIZE)

dataset = tf.data.Dataset.zip((dataset_a, dataset_b))
dataset = dataset.prefetch(2)
# endregion


# ==================================================================================================================== #
# Load model
# ==================================================================================================================== #
# region

strategy = tf.distribute.MirroredStrategy()
print('Number of devices:', strategy.num_replicas_in_sync)

with strategy.scope():
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

    lr_decay_fn = lr_scheduler.WarmUpCosine(
        learning_rate_base=1e-3,
        total_steps=EPOCHS * STEPS_PER_EPOCH,
        warmup_learning_rate=0.0,
        warmup_steps=WARMUP_STEPS
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay_fn)


    # Get model
    resnet_enc.trainable = True
    barlow_twins = BarlowTwins(resnet_enc)
    barlow_twins.compile(optimizer=optimizer)
# endregion


# ==================================================================================================================== #
# Train model
# ==================================================================================================================== #
# region

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

# For performance analysis
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard = tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, profile_batch='0,2867')

history = barlow_twins.fit(
    dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[es, mc, tboard]
)

with open('trained_models/resnet_classifiers/1024/history.pickle', 'wb') as file:
    pickle.dump(history.history, file)
# endregion
