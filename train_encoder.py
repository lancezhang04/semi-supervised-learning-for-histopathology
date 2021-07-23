from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from utils.image_augmentation import get_blur_layer
from utils.train.callbacks import EncoderCheckpoint
from utils.models.barlow_twins import BarlowTwins
from utils.datasets import get_dataset_df
from utils import image_augmentation
from utils.train.lr_scheduler import get_decay_fn
from utils.models import resnet_cifar, resnet

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
        model_name = f'encoder_{dataset_type}_{IMAGE_SHAPE[0]}_{PROJECTOR_DIM}_' + \
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

    print('Dataset length:', len(df))

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


def load_model(steps_per_epoch, cifar_resnet=True):
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices:', strategy.num_replicas_in_sync)

    with strategy.scope():
        if cifar_resnet:
            resnet_enc = resnet_cifar.get_network(
                n=2,
                hidden_dim=PROJECTOR_DIM,
                use_pred=False,
                return_before_head=False,
                input_shape=IMAGE_SHAPE
            )
            if MODEL_WEIGHTS:
                resnet_enc.load_weights(MODEL_WEIGHTS)
                if VERBOSE:
                    print('Using (pretrained) model weights')
        else:
            resnet_enc = resnet.get_barlow_encoder(IMAGE_SHAPE, PROJECTOR_DIM, hidden_layers=3)

        blur_layer = get_blur_layer(FILTER_SIZE, IMAGE_SHAPE)

        # Load optimizer
        lr_decay_fn = get_decay_fn(LEARNING_RATE_BASE, EPOCHS, steps_per_epoch)
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


def main(suffix=None, model_name=None, cifar_resnet=True):
    save_dir = configure_saving(suffix, model_name)
    print('Saving at:', save_dir)

    dataset, steps_per_epoch = load_dataset()
    barlow_twins, resnet_enc = load_model(steps_per_epoch, cifar_resnet=cifar_resnet)

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
        
    # Save weights for the ResNet backbone
    resnet_enc.layers[1].save_weights(os.path.join(save_dir, 'resnet.h5'))


if __name__ == '__main__':
    # Overwrite default values
    BATCH_SIZE = 256
    IMAGE_SHAPE = [224, 224, 3]
    EPOCHS = 30
    ROOT_SAVE_DIR = 'trained_models/encoders'
    
    PROJECTOR_DIM = 2048
    main(model_name='encoder_resnet50_2048', cifar_resnet=False)

#     PROJECTOR_DIMENSIONALITY = 512
#     main(model_name='encoder_512')

#     PROJECTOR_DIMENSIONALITY = 2048
#     main(model_name='encoder_2048')

#     PROJECTOR_DIMENSIONALITY = 4096
#     main(model_name='encoder_4096')

#     PROJECTOR_DIMENSIONALITY = 8192
#     main(model_name='encoder_8192')

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
