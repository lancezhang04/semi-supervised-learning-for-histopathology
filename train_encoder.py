import numpy as np
import yaml
import json
import os

# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_addons as tfa
import tensorflow as tf

from utils.image_augmentation import get_blur_layer
from utils.train.callbacks import EncoderCheckpoint, BTDebug
from utils.models.barlow_twins import BarlowTwins
from utils.datasets import get_dataset_df
from utils import image_augmentation
from utils.train.lr_scheduler import get_decay_fn
from utils.models import resnet_cifar, resnet
from utils.misc import log_config


def configure_saving(model_name):
    print('Model name:', model_name)

    save_dir = os.path.join(config['root_save_dir'], model_name)
    config['save_dir'] = save_dir

    try:
        os.makedirs(save_dir, exist_ok=False)
    except FileExistsError:
        input_ = input('save_dir already exists, continue? (Y/n)  >> ')
        if input_ != 'Y':
            raise ValueError

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    return save_dir


def load_dataset():
    # Only using training set (and no validation set)
    df = get_dataset_df(config['dataset_config'], config['random_seed'], mode='encoder')
    # Shuffle the dataset (NumPy random seed)
    df = df.sample(frac=1).reset_index(drop=True)

    print('Dataset length:', len(df))

    # Generate one dataset for each view
    datagens = []
    for i in range(2):
        datagen = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
            df[df['split'] == 'train'],
            shuffle=False,
            seed=config['random_seed'],
            target_size=config['image_shape'][:2],
            batch_size=config['batch_size']
        )
        datagens.append(datagen)
    
    # There seems to be some issues with creating the datasets in a loop, which results in the two
    # datasets referencing the same ImageDataGenerator, so I decided to just leave it be
    ds_a = tf.data.Dataset.from_generator(
        lambda: [datagens[0].next()[0]],
        output_types='float32', output_shapes=[None] * 4
    )
    ds_a = ds_a.map(
        lambda x: image_augmentation.augment(
            x, view=0, 
            config=config['preprocessing_config']
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # Some tf image functions return values that are not in the range of [0, 1]
    ds_a = ds_a.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_b = tf.data.Dataset.from_generator(
        lambda: [datagens[1].next()[0]],
        output_types='float32', output_shapes=[None] * 4
    )
    ds_b = ds_b.map(
        lambda x: image_augmentation.augment(
            x, view=1, 
            config=config['preprocessing_config']
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_b = ds_b.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
    datasets = [ds_a, ds_b]

    # Combine the two datasets, input must be tuple
    dataset = tf.data.Dataset.zip(tuple(datasets))
    dataset = dataset.prefetch(config['prefetch'])

    # This creates a generator from the dataset
    def get_generator():
        while True:
            yield next(iter(dataset))

    steps_per_epoch = len(datagens[0])
    config['steps_per_epoch'] = steps_per_epoch
    print('Steps per epoch:', steps_per_epoch)

    return get_generator(), datagens


def load_model():
    strategy = tf.distribute.MirroredStrategy(config['gpu_used'])
    print('Number of devices:', strategy.num_replicas_in_sync)

    with strategy.scope():
        if config['cifar_resnet']:
            resnet_enc = resnet_cifar.get_network(
                n=2,
                hidden_dim=config['projector_dim'],
                use_pred=False,
                return_before_head=False,
                input_shape=config['image_shape'])
        else:
            resnet_enc = resnet.get_barlow_encoder(config['image_shape'], config['projector_dim'], hidden_layers=3)

        blur_layer = get_blur_layer(config['filter_size'], config['image_shape'])

        # Load optimizer
        lr_decay_fn = get_decay_fn(config['lr_base'], config['epochs'], config['steps_per_epoch'])

        if config['use_lamb']:
            optimizer = tfa.optimizers.LAMB(
                learning_rate=lr_decay_fn,
                weight_decay_rate=config['weight_decay']
            )
        else:
            optimizer = tf.keras.optimizers.Adam(lr_decay_fn)

        # Get model
        barlow_twins = BarlowTwins(
            resnet_enc, blur_layer=blur_layer,
            preprocessing_config=config['preprocessing_config'],
            batch_size=config['batch_size']
        )
        barlow_twins.compile(optimizer=optimizer)

        print('Barlow twins blur probabilities:', barlow_twins.blur_probabilities)

    # This is necessary to prevent TensorFlow from raising an exception
    barlow_twins.built = True
    barlow_twins.summary()

    return barlow_twins, resnet_enc


def main(model_name=None):
    save_dir = configure_saving(model_name)
    print('Saving at:', save_dir)

    dataset, datagens = load_dataset()
    barlow_twins, resnet_enc = load_model()

    callbacks = []
    if config['patience'] is not None:
        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=config['patience'])
        callbacks.append(es)
        
    mc = EncoderCheckpoint(resnet_enc, save_dir)
    callbacks.append(mc)
    
    debug = BTDebug(datagens)
    callbacks.append(debug)
    
    log_config(config, save_config=True)
    history = barlow_twins.fit(
        dataset,
        epochs=config['epochs'],
        steps_per_epoch=config['steps_per_epoch'],
        callbacks=callbacks
    )

    # Save training history
    with open(os.path.join(save_dir, 'history.json'), 'wb') as f:
        json.dump(history.history, f)

    # Save weights for the best ResNet backbone
    resnet_enc.load_weights(os.path.join(save_dir, 'encoder.h5'))
    resnet_enc.layers[1].save_weights(os.path.join(save_dir, 'resnet.h5'))


if __name__ == '__main__':
    with open('config/encoder_config.yaml') as file:
        config = yaml.safe_load(file)
    
    # Adjust learning rate according to the batch size
    config['lr_base'] = config['lr_base'] * config['batch_size'] / 256
    config['epochs'] = 100
    
    # Set random seed
    np.random.seed(config['random_seed'])
    tf.random.set_seed(config['random_seed'])
    
    # Training is getting stuck after the second epoch... can't figure out why!
    config['use_lamb'] = True
    config['batch_size'] = 64 # Maybe reducing the batch size will help?
    main(model_name='encoder_resnet50_test')
