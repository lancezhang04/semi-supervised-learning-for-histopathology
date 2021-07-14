from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.models import Model

from utils.image_augmentation import get_gaussian_filter
from utils import image_augmentation
from utils.train import lr_scheduler
from utils.train.callbacks import EncoderCheckpoint
from utils.models import resnet
from utils.models.barlow_twins import BarlowTwins
from utils.datasets import get_dataset_df, create_encoder_dataset

from optparse import OptionParser
import numpy as np
import pickle
import json
import os


parser = OptionParser()
parser.add_option('-s', '--suffix', type='string', default='')
parser.add_option('--root-save-dir', dest='root_save_dir', type='string', default='trained_models/encoders')
parser.add_option('--no-blur', dest='blur', default=True, action='store_false')
parser.add_option('--no-color', dest='color', default=True, action='store_false')
(options, args) = parser.parse_args()

# ==================================================================================================================== #
# Configuration
# ==================================================================================================================== #
# region

VERBOSE = 1
PATIENCE = 30
EPOCHS = 10
BATCH_SIZE = 256
PREFETCH = 6

IMAGE_SHAPE = [224, 224, 3]
FILTER_SIZE = 23

PROJECTOR_DIMENSIONALITY = 1024
LEARNING_RATE_BASE = 5e-4

PREPROCESSING_CONFIG = {
    'vertical_flip_probability': 0.5,
    'color_jittering': 0.8,
    'color_dropping_probability': 0.2,
    'brightness_adjustment_max_intensity': 0.4,
    'contrast_adjustment_max_intensity': 0.4,
    'color_adjustment_max_intensity': 0.2,
    'hue_adjustment_max_intensity': 0.1,
    'gaussian_blurring_probability': [1, 0.1],
    'solarization_probability': [0, 0.2]
}

MODEL_WEIGHTS = None  # if continuing training
ROOT_SAVE_DIR = options.root_save_dir  # base directory to save at

DATASET_CONFIG = {
    'split': 'tissue_classification/fold_test.csv',
    'train_split': 1,
    'validation_split': 0,
    'dataset_dir': 'tissue_classification/tissue_classification',
    'groups': {},
    'major_groups': []
}

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
# tf.random.set_seed(RANDOM_SEED)  # This might be messing up augmentation

if not options.blur:
    print('Not performing Gaussian blurring...')
    PREPROCESSING_CONFIG['gaussian_blurring_probability'] = [0, 0]
if not options.color:
    print('Not performing color changes (jittering & solarization)...')
    PREPROCESSING_CONFIG['color_jittering'] = 0
    PREPROCESSING_CONFIG['solarization_probability'] = [0, 0]
# endregion


# ==================================================================================================================== #
# Saving information
# ==================================================================================================================== #
# region

dataset_type = 'tissue' if 'tissue' in DATASET_CONFIG['dataset_dir'] else 'cell'
model_name = f'encoder_{dataset_type}_{IMAGE_SHAPE[0]}_{PROJECTOR_DIMENSIONALITY}_' + \
             f'{BATCH_SIZE}_{EPOCHS}_{LEARNING_RATE_BASE}' + options.suffix
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

datagen_a = ImageDataGenerator(rescale=1./225).flow_from_dataframe(
df[df['split'] == 'train'],
    seed=RANDOM_SEED,
    target_size=IMAGE_SHAPE[:2], batch_size=1
)

datagen_b = ImageDataGenerator(rescale=1./225).flow_from_dataframe(
df[df['split'] == 'train'],
    seed=RANDOM_SEED,
    target_size=IMAGE_SHAPE[:2], batch_size=1
)


ds_a = create_encoder_dataset(datagen_a)
ds_a = ds_a.map(
    lambda x: image_augmentation.augment(x, 0, FILTER_SIZE, config=PREPROCESSING_CONFIG),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_a = ds_a.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=tf.data.experimental.AUTOTUNE)

"""CHANGE THIS LATER"""
ds_b = create_encoder_dataset(datagen_b)
ds_b = ds_b.map(
    lambda x: image_augmentation.augment(x, 1, FILTER_SIZE, config=PREPROCESSING_CONFIG),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_b = ds_b.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = tf.data.Dataset.zip((ds_a, ds_b))
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(PREFETCH)


STEPS_PER_EPOCH = len(datagen_a) // BATCH_SIZE
TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
# endregion


# ==================================================================================================================== #
# Load model
# ==================================================================================================================== #
# region

strategy = tf.distribute.MirroredStrategy()
print('Number of devices:', strategy.num_replicas_in_sync)

with strategy.scope():
    # -------------------
    # Model      | n   |
    # ResNet20   | 2   |
    # ResNet56   | 6   |
    # ResNet110  | 12  |
    # ResNet164  | 18  |
    # ResNet1001 | 111 |

    resnet_enc = resnet.get_network(
        n=2,
        hidden_dim=PROJECTOR_DIMENSIONALITY,
        use_pred=False,
        return_before_head=False,
        input_shape=IMAGE_SHAPE
    )
    if MODEL_WEIGHTS:
        resnet_enc.load_weights(MODEL_WEIGHTS)
        if VERBOSE:
            print('Using (pretrained) model weights')

    kernel_weights = get_gaussian_filter((FILTER_SIZE, FILTER_SIZE), sigma=1)
    in_channels = 3
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1)
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    blur_layer = DepthwiseConv2D(FILTER_SIZE, use_bias=False, padding='same')

    inputs = Input(IMAGE_SHAPE)
    outputs = blur_layer(inputs)
    blur_layer = Model(inputs=inputs, outputs=outputs)

    blur_layer.layers[1].set_weights([kernel_weights])
    blur_layer.trainable = False

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

    # from utils.train.callbacks import LRFinder
    # lr_finder = LRFinder(
    #     min_lr=1e-9,
    #     max_lr=1e-1,
    #     steps_per_epoch=STEPS_PER_EPOCH,
    #     epochs=EPOCHS
    # )

    # Get model
    barlow_twins = BarlowTwins(
            resnet_enc, blur_layer=blur_layer, 
            preprocessing_config=PREPROCESSING_CONFIG,
            batch_size=BATCH_SIZE
    )
    barlow_twins.compile(optimizer=optimizer)
    print('Barlow twins blur probabilities:', barlow_twins.blur_probabilities)
# endregion


# ==================================================================================================================== #
# Train model
# ==================================================================================================================== #
# region

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=PATIENCE)
mc = EncoderCheckpoint(resnet_enc, SAVE_DIR)

# For performance analysis
# logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tboard = tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, profile_batch='0,2867')

history = barlow_twins.fit(
    dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[es, mc]
)

with open(os.path.join(SAVE_DIR, 'history.pickle'), 'wb') as file:
    pickle.dump(history.history, file)

lr_finder.plot_lr()
lr_finder.plot_loss()
# endregion
