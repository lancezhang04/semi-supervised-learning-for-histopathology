from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from utils import image_augmentation
from utils.train import lr_scheduler
from utils.models import resnet20
from utils.models.barlow_twins import BarlowTwins
from utils.datasets import get_dataset_df, create_encoder_dataset
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import pickle
import json
import os


# ==================================================================================================================== #
# Configuration
# ==================================================================================================================== #
# region

VERBOSE = 1
BATCH_SIZE = 256
PATIENCE = 30
EPOCHS = 5
IMAGE_SHAPE = [224, 224, 3]
FILTER_SIZE = 3
PROJECTOR_DIMENSIONALITY = 4096
LEARNING_RATE_BASE = 1e-3

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

MODEL_WEIGHTS = None  # if continuing training
ROOT_SAVE_DIR = 'trained_models/encoders'  # base directory to save at

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
# Saving information
# ==================================================================================================================== #
# region

dataset_type = 'tissue' if 'tissue' in DATASET_CONFIG['dataset_dir'] else 'cell'
model_name = f'encoder_{dataset_type}_{IMAGE_SHAPE[0]}_{PROJECTOR_DIMENSIONALITY}_' + \
             f'{BATCH_SIZE}_{EPOCHS}_{LEARNING_RATE_BASE}'
print('Model name:', model_name)

SAVE_DIR = os.path.join(ROOT_SAVE_DIR, model_name)

try:
    os.makedirs(SAVE_DIR, exist_ok=False)
except:
    input_ = input('save_dir already exists, continue? (Y/n)  >> ')
    if input_ != 'Y':
        raise ValueError

with open(os.path.join(SAVE_DIR, 'preprocessing_config.json'), 'w') as file:
    json.dump(PREPROCESSING_CONFIG, file, indent=4)

with open(os.path.join(SAVE_DIR, 'dataset_config.json'), 'w') as file:
    json.dump(DATASET_CONFIG, file, indent=4)
# endregion


# ==================================================================================================================== #
# Load data generators
# P.S. Current implementation is a little hacky
# ==================================================================================================================== #
# region

# Only using training set (and no validation set)
df = get_dataset_df(DATASET_CONFIG, RANDOM_SEED)

datagen = ImageDataGenerator(rescale=1./225).flow_from_dataframe(
df[df['split'] == 'train'],
    seed=RANDOM_SEED,
    target_size=IMAGE_SHAPE[:2], batch_size=1
)


ds_a = create_encoder_dataset(datagen)
ds_a = ds_a.map(
    lambda x: image_augmentation.augment(x, 0, FILTER_SIZE),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_a = ds_a.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=tf.data.experimental.AUTOTUNE)

ds_b = create_encoder_dataset(datagen)
ds_b = ds_b.map(
    lambda x: image_augmentation.augment(x, 1, FILTER_SIZE),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_b = ds_b.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = tf.data.Dataset.zip((ds_a, ds_b))
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(40)


STEPS_PER_EPOCH = len(datagen) // BATCH_SIZE
TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
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
        learning_rate_base=LEARNING_RATE_BASE,
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
        self.save_dir = os.path.join(SAVE_DIR, 'encoder.h5')
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

with open(os.path.join(SAVE_DIR, 'history.pickle', 'wb')) as file:
    pickle.dump(history.history, file)
# endregion
