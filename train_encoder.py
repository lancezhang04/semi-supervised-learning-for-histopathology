from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
import tensorflow as tf

from utils.image_augmentation import get_gaussian_filter
from utils.train.callbacks import EncoderCheckpoint
from utils.models.barlow_twins import BarlowTwins
from utils.datasets import get_dataset_df
from utils import image_augmentation
from utils.train import lr_scheduler
from utils.models import resnet

import numpy as np
import pickle
import json
import os

# Default configuration is stored in here
from config.encoder_default_config import *
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def configure_saving(suffix=None, model_name=None):
    # Format model name
    if model_name is None:
        dataset_type = 'tissue' if 'tissue' in DATASET_CONFIG['dataset_dir'] else 'cell'
        model_name = f'encoder_{dataset_type}_{IMAGE_SHAPE[0]}_{PROJECTOR_DIMENSIONALITY}_' + \
                     f'{BATCH_SIZE}_{EPOCHS}_{LEARNING_RATE_BASE}'
        if suffix is not None:
            suffix = '_' + suffix if suffix[0] != '_' else suffix
            model_name += suffix
    print('Model name:', model_name)

    save_dir = os.path.join(ROOT_SAVE_DIR, model_name)

    try:
        os.makedirs(save_dir, exist_ok=False)
    except FileExistsError:
        input_ = input('save_dir already exists, continue? (Y/n)  >> ')
        if input_ != 'Y':
            raise ValueError

    with open(os.path.join(save_dir, 'preprocessing_config.json'), 'w') as file:
        json.dump(PREPROCESSING_CONFIG, file, indent=4)

    with open(os.path.join(save_dir, 'dataset_config.json'), 'w') as file:
        json.dump(DATASET_CONFIG, file, indent=4)

    return save_dir


def load_dataset():
    # Only using training set (and no validation set)
    df = get_dataset_df(DATASET_CONFIG, RANDOM_SEED, mode='encoder')
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    datagen_a = ImageDataGenerator(rescale=1. / 225).flow_from_dataframe(
        df[df['split'] == 'train'],
        shuffle=False,
        seed=RANDOM_SEED,
        target_size=IMAGE_SHAPE[:2], batch_size=BATCH_SIZE
    )

    datagen_b = ImageDataGenerator(rescale=1. / 225).flow_from_dataframe(
        df[df['split'] == 'train'],
        shuffle=False,
        seed=RANDOM_SEED,
        target_size=IMAGE_SHAPE[:2], batch_size=BATCH_SIZE
    )

    ds_a = tf.data.Dataset.from_generator(lambda: [datagen_a.next()[0]], output_types='float32',
                                          output_shapes=[None] * 4)
    ds_a = ds_a.map(
        lambda x: image_augmentation.augment(x, 0, config=PREPROCESSING_CONFIG),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_a = ds_a.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_b = tf.data.Dataset.from_generator(lambda: [datagen_b.next()[0]], output_types='float32',
                                          output_shapes=[None] * 4)
    ds_b = ds_b.map(
        lambda x: image_augmentation.augment(x, 1, config=PREPROCESSING_CONFIG),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_b = ds_b.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = tf.data.Dataset.zip((ds_a, ds_b))
    # dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(PREFETCH)

    # This creates a generator from the dataset
    def data_generator():
        while True:
            yield next(iter(dataset))

    steps_per_epoch = len(datagen_a)
    print('Steps per epoch:', steps_per_epoch)

    return data_generator(), steps_per_epoch


def load_model(steps_per_epoch):
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
        WARMUP_STEPS = int(WARMUP_EPOCHS * steps_per_epoch)

        lr_decay_fn = lr_scheduler.WarmUpCosine(
            learning_rate_base=LEARNING_RATE_BASE,
            total_steps=EPOCHS * steps_per_epoch,
            warmup_learning_rate=0.0,
            warmup_steps=WARMUP_STEPS
        )
        optimizer = tf.keras.optimizers.Adam(lr_decay_fn)

        # Get model
        barlow_twins = BarlowTwins(
            resnet_enc, blur_layer=blur_layer,
            preprocessing_config=PREPROCESSING_CONFIG,
            batch_size=BATCH_SIZE
        )
        barlow_twins.compile(optimizer=optimizer)

        print('Barlow twins blur probabilities:', barlow_twins.blur_probabilities)

    # This is necessary to prevent TensorFlow from raising an exception
    barlow_twins.built = True
    barlow_twins.summary()

    return barlow_twins, resnet_enc


def main(suffix=None, model_name=None):
    save_dir = configure_saving(suffix, model_name)
    print('Saving at:', save_dir)

    dataset, steps_per_epoch = load_dataset()
    barlow_twins, resnet_enc = load_model(steps_per_epoch)

    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=PATIENCE)
    mc = EncoderCheckpoint(resnet_enc, save_dir)

    print('\nSteps per epoch:', steps_per_epoch)
    history = barlow_twins.fit(
        dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=[es, mc]
    )

    # Save training history
    with open(os.path.join(save_dir, 'history.pickle'), 'wb') as file:
        pickle.dump(history.history, file)


if __name__ == '__main__':
    # Overwrite default values
    BATCH_SIZE = 32
    IMAGE_SHAPE = [32, 32, 3]
    PROJECTOR_DIMENSIONALITY = 512
    EPOCHS = 1
    ROOT_SAVE_DIR = 'trained_models/encoders/0715'

    main(model_name='test')

    # PREPROCESSING_CONFIG['color_jittering'] = 0
    # main('no_color')
    #
    # PREPROCESSING_CONFIG['color_jittering'] = 0.8
    # PREPROCESSING_CONFIG['gaussian_blurring_probability'] = [0, 0]
    # main('no_blur')
    #
    # PREPROCESSING_CONFIG['color_jittering'] = 0
    # main('flip_only')
    #
    # PREPROCESSING_CONFIG['horizontal_flip'] = False
    # PREPROCESSING_CONFIG['vertical_flip'] = False
    # main('no_aug')