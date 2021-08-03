import numpy as np
import pickle
import json
import os
import yaml

# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
from tensorflow_addons.optimizers import SGDW, MultiOptimizer
import tensorflow as tf

from utils.datasets import get_generators, create_classifier_dataset
from utils.train import lr_scheduler
from utils.models import resnet_cifar, resnet, vae
from utils.misc import log_config
from config.datasets_config import DATASETS_CONFIG


def configure_saving():
    # Generate save directory and store in config
    save_dir = os.path.join(config['root_save_dir'], config['model_name'])
    config['save_dir'] = save_dir

    # Create save directory (if it does not exist)
    try:
        os.makedirs(save_dir, exist_ok=False)
    except FileExistsError:
        input_ = input('save_dir already exists, continue? (Y/n)  >> ')
        if input_ != 'Y':
            raise ValueError

    return save_dir


def load_datasets():
    dataset_config['train_split'] = config['train_split']
    dataset_config['validation_split'] = config['validation_split']

    # Load data generators
    datagen, datagen_val, datagen_test = get_generators(
        ['train', 'val', 'test'],
        config['image_shape'],
        batch_size=1,  # batched later
        random_seed=config['random_seed'],
        dataset_config=dataset_config
    )
    classes = list(datagen.class_indices.keys())
    config['classes'] = classes
    config['num_classes'] = len(classes)

    # Load class weight
    class_weight = None
    if config['use_class_weight']:
        with open(os.path.join(dataset_config['dataset_dir'], 'class_weight.json'), 'r') as f:
            class_weight = json.load(f)
        groups = dataset_config['groups']
        class_weight = {groups[k]: v for k, v in class_weight.items() if k in groups.keys()}
        class_weight = {datagen.class_indices[k]: v for k, v in class_weight.items()}
        print('Using class weights:', class_weight)
    config['class_weight'] = class_weight

    # Load datasets
    datasets, steps = [], []
    for gen in [datagen, datagen_val, datagen_test]:
        ds = create_classifier_dataset(gen, config['image_shape'], len(classes))
        ds = ds.batch(config['batch_size'])
        ds = ds.prefetch(config['prefetch'])

        steps.append(len(gen) // config['batch_size'])
        datasets.append(ds)
    config['steps'] = steps

    return datasets


def load_model(config_dict, evaluation=False):
    """
    @param config_dict:     the configuration for the model
    @param evaluation:      whether or not the model is loaded for testing
    @return:                classification model
    """
    strategy = tf.distribute.MirroredStrategy(config_dict['gpu_used'])
    print('Number of devices:', strategy.num_replicas_in_sync)

    with strategy.scope():
        model_type = config_dict['model_type']
        if model_type not in ['supervised', 'barlow', 'barlow_fine_tuned']:
            raise ValueError
        encoder_trainable = False if (model_type.startswith('barlow') and 'fine_tuned' not in model_type) else True
        print('Encoder trainable:', encoder_trainable)

        # Determine path to encoder weights
        if evaluation or model_type == 'supervised':
            # Test time ==> no need to load pre-trained weights
            encoder_weights_path = None
        elif model_type.startswith('barlow'):
            if config_dict['encoder_type'] == 'resnet':
                model_name = 'resnet.h5'
            else:
                model_name = 'encoder.h5'

            encoder_weights_path = os.path.join(config_dict['pretrained_dir'], model_name)
            print('Loading encoder weights from:', encoder_weights_path)
        else:
            raise ValueError

        # Load model and initialize weights
        if config_dict['model_type'] == 'cifar':
            # Smaller modified ResNet20 that outputs a 256-d features?
            model = resnet_cifar.get_classifier(
                projector_dim=config_dict['projector_dim'],
                num_classes=config_dict['num_classes'],
                encoder_weights=encoder_weights_path,
                image_shape=config_dict['image_shape']
            )
        elif config_dict['model_type'] == 'resnet50':
            # Updated (larger) version of the encoder (ResNet50v2), ~20M parameters
            model = resnet.get_classifier(
                num_classes=config_dict['num_classes'],
                input_shape=config_dict['image_shape'],
                encoder_weights=encoder_weights_path,
                encoder_trainable=encoder_trainable
            )
        elif config_dict['model_type'] == 'vae':
            # Encoder of a variational autoencoder
            model = vae.get_classifier(
                config=config_dict,
                encoder_weights_path=encoder_weights_path
            )
        else:
            raise ValueError

        # Set up optimizer
        warmup_epochs = 0.1
        warmup_steps = int(warmup_epochs * config_dict['steps_per_epoch'])

        def get_optimizer(base_lr):
            lr_fn = lr_scheduler.WarmUpCosine(
                learning_rate_base=base_lr,
                total_steps=config_dict['epochs'] * config_dict['steps_per_epoch'],
                warmup_learning_rate=0.0,
                warmup_steps=warmup_steps
            )
            return SGDW(learning_rate=lr_fn, momentum=0.9, weight_decay=0)

        optimizers_and_layers = [
            (get_optimizer(config_dict['encoder_lr']), model.layers[1]),  # encoder
            (get_optimizer(config_dict['head_lr']), model.layers[2])  # classification head
        ]
        optimizer = MultiOptimizer(optimizers_and_layers)

        # Print model summary and compile model
        model.summary()
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'acc',
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
                MatthewsCorrelationCoefficient(num_classes=config_dict['num_classes'], name='MCC')
            ]
        )

    return model


def main(model_name=None):
    config['model_name'] = model_name
    configure_saving()

    # Load dataset and model
    datasets = load_datasets()
    model = load_model(config_dict=config)

    # Create training callbacks
    callbacks = []
    if config['patience'] is not None:
        es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=config['patience'])
        callbacks.append(es)

    mc = ModelCheckpoint(
        os.path.join(config['save_dir'], 'classifier.h5'),
        monitor='val_acc', mode='max',
        verbose=1,
        save_best_only=True, save_weights_only=True
    )
    callbacks.append(mc)

    # Print and save the configuration
    log_config(config, save_config=True)

    # Train the model
    history = model.fit(
        datasets[0],
        epochs=config['epochs'],
        steps_per_epoch=config['steps'][0],
        validation_steps=config['steps'][1],
        validation_data=datasets[1],
        callbacks=callbacks,
        class_weight=config['class_weight']
    )

    # Save the training history
    with open(os.path.join(config['save_dir'], 'history.pickle'), 'wb') as f:
        pickle.dump(history.history, f)

    # Load best model, save encoder weights (separately), and evaluate model
    model.load_weights(os.path.join(config['save_dir'], 'classifier.h5'))
    model.layers[1].save_weights(os.path.join(config['save_dir'], 'encoder.h5'))
    model.evaluate(datasets[2], steps=config['steps'][2])


if __name__ == '__main__':
    with open('config/classifier_config.yaml') as file:
        config = yaml.safe_load(file)

    dataset_config = DATASETS_CONFIG[config['dataset_type']]

    np.random.seed(config['random_seed'])
    tf.random.set_seed(config['random_seed'])

    config['model_type'] = 'barlow_fine_tuned'
    config['pretrained_dir'] = 'trained_models/encoders/encoder_resnet50_100_baseline'

    main(model_name=f'{config["model_type"]}_{config["head_lr"]}_{config["encoder_lr"]}')
