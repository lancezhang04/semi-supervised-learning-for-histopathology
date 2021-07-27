from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import numpy as np
import pickle
import json
import os
import yaml

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
from tensorflow.keras.layers import Input, Dense
from tensorflow_addons.optimizers import SGDW, MultiOptimizer
from tensorflow.keras.models import Model
import tensorflow as tf

from utils.datasets import get_generators, create_classifier_dataset
from utils.train import lr_scheduler
from utils.models import resnet_cifar, resnet
from config.datasets_config import DATASETS_CONFIG


def configure_saving(model_name=None):
    if model_name is None:
        raise ValueError('Automatic model name generation deprecated')
    print('Model name:', model_name)

    save_dir = os.path.join(config['root_save_dir'], model_name)

    try:
        os.makedirs(save_dir, exist_ok=False)
    except FileExistsError:
        input_ = input('save_dir already exists, continue? (Y/n)  >> ')
        if input_ != 'Y':
            raise ValueError

    with open(os.path.join(save_dir, 'dataset_config.json'), 'w') as file:
        json.dump(dataset_config, file, indent=4)

    return save_dir


def load_datasets():
    # Load data generators
    datagen, datagen_val, datagen_test = get_generators(
        ['train', 'val', 'test'],
        config['image_shape'], 1,
        config['random_seed'],
        dataset_config=dataset_config
    )
    classes = list(datagen.class_indices.keys())
    print(datagen.class_indices)
    
    # Load class weight
    class_weight = None
    if config['use_class_weight']:
        with open(os.path.join(dataset_config['dataset_dir'], 'class_weight.json'), 'r') as file:
            class_weight = json.load(file)
        groups = dataset_config['groups']
        class_weight = {groups[k]: v for k, v in class_weight.items() if k in groups.keys()}
        class_weight = {datagen.class_indices[k]: v for k, v in class_weight.items()}
        print('Using class weights:', class_weight)

    # Load datasets
    datasets = []
    steps = []

    for gen in [datagen, datagen_val, datagen_test]:
        steps.append(len(gen) // config['batch_size'])

        ds = create_classifier_dataset(gen, config['image_shape'], len(classes))
        ds = ds.batch(config['batch_size'])
        ds = ds.prefetch(config['prefetch'])
        datasets.append(ds)

    return datasets, steps, classes, class_weight


def load_model(config, evaluation=False):

    strategy = tf.distribute.MirroredStrategy(config['gpu_used'])
    print('Number of devices:', strategy.num_replicas_in_sync)

    with strategy.scope():
        model_type = config['model_type']
        if model_type not in ['supervised', 'barlow', 'barlow_fine_tuned']:
            raise ValueError
        encoder_trainable = False if ('barlow' in model_type and 'fine_tuned' not in model_type) else True
        print('Encoder trainable:', encoder_trainable)

        if evaluation:
            # Test time ==> no need to load pre-trained weights
            encoder_weights_path = None
        elif 'barlow' in model_type:
            model_name = 'encoder.h5' if config['cifar_resnet'] else 'resnet.h5'
            encoder_weights_path = os.path.join(config['pretrained_dir'], model_name)
            print('Loading encoder weights from:', encoder_weights_path)
        else:
            encoder_weights_path = None

        if config['cifar_resnet']:
            # Smaller modified ResNet20 that outputs a 256-d features?
            model = resnet_cifar.get_classifier(
                projector_dim=config['projector_dim'],
                num_classes=config['num_classes'],
                encoder_weights=encoder_weights_path,
                image_shape=config['image_shape']
            )
        else:
            # Updated (larger) version of the encoder (ResNet50v2)
            model = resnet.get_classifier(
                num_classes=config['num_classes'],
                input_shape=config['image_shape'],
                encoder_weights=encoder_weights_path,
                encoder_trainable=encoder_trainable
            )

        # Set up optimizer
        warmup_epochs = 0.1
        warmup_steps = int(warmup_epochs * config['steps_per_epoch'])
        
        def get_optimizer(base_lr):
            lr_fn = lr_scheduler.WarmUpCosine(
                learning_rate_base=base_lr,
                total_steps=config['epochs'] * config['steps_per_epoch'],
                warmup_learning_rate=0.0,
                warmup_steps=warmup_steps
            )
            
            return SGDW(learning_rate=lr_fn, momentum=0.9, weight_decay=0)
        
        print('Using learning rates:', config['head_lr'], config['encoder_lr'])
        optimizers_and_layers = [
            (get_optimizer(config['encoder_lr']), model.layers[1]),  # encoder
            (get_optimizer(config['head_lr']), model.layers[2])  # classification head
        ]
        optimizer = MultiOptimizer(optimizers_and_layers)
        
        # Compile model
        model.summary()
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'acc',
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
                MatthewsCorrelationCoefficient(num_classes=config['num_classes'], name='MCC')
            ]
        )

    return model


def main(model_name=None):
    save_dir = configure_saving(model_name)
    print('Saving at:', save_dir)

    datasets, steps, classes, class_weight = load_datasets()
    steps_per_epoch, validation_steps, test_steps = steps

    config['num_classes'] = len(classes)
    config['steps_per_epoch'] = steps_per_epoch

    model = load_model(config=config)

    callbacks = []
    if config['patience'] is not None:
        es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=config['patience'])
        callbacks.append(es)

    mc = ModelCheckpoint(
        os.path.join(save_dir, 'classifier.h5'),
        monitor='val_acc', mode='max',
        verbose=1,
        save_best_only=True, save_weights_only=True
    )
    callbacks.append(mc)

    print('\nSteps per epoch:', steps_per_epoch)
    history = model.fit(
        datasets[0],
        epochs=config['epochs'],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_data=datasets[1],
        callbacks=callbacks,
        class_weight=class_weight
    )
    with open(os.path.join(save_dir, 'history.pickle'), 'wb') as file:
        pickle.dump(history.history, file)

    model.load_weights(os.path.join(save_dir, 'classifier.h5'))
    model.layers[1].save_weights(os.path.join(save_dir, 'encoder.h5'))

    model.evaluate(datasets[2], steps=test_steps)


if __name__ == '__main__':
    with open('config/classifier_config.yaml') as file:
        config = yaml.safe_load(file)
        print(config)

    dataset_config = DATASETS_CONFIG[config['dataset_type']]

    np.random.seed(config['random_seed'])
    tf.random.set_seed(config['random_seed'])

    main(model_name=f'{config["model_type"]}_{config["head_lr"]}_{config["encoder_lr"]}')
