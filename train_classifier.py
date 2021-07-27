from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
from tensorflow.keras.layers import Input, Dense
from tensorflow_addons.optimizers import SGDW
from tensorflow.keras.models import Model
import tensorflow as tf

from utils.datasets import get_generators, create_classifier_dataset
from utils.train import lr_scheduler
from utils.models import resnet_cifar, resnet


import numpy as np
import pickle
import json
import os

from config.classifier_default_config import *
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def configure_saving(suffix=None, model_name=None):
    if model_name is None:
        dataset_type = DATASET_CONFIG['type']
        model_name = f'{MODEL_TYPE}_' + \
                     f'{dataset_type}_{IMAGE_SHAPE[0]}_{DATASET_CONFIG["train_split"]}_' + \
                     f'{PROJECTOR_DIM}_' + \
                     f'{BATCH_SIZE}_{EPOCHS}_{LEARNING_RATE}'
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

    with open(os.path.join(save_dir, 'dataset_config.json'), 'w') as file:
        json.dump(DATASET_CONFIG, file, indent=4)

    return save_dir


def load_datasets():
    datagen, datagen_val, datagen_test = get_generators(
        ['train', 'val', 'test'],
        IMAGE_SHAPE, 1,
        RANDOM_SEED, config=DATASET_CONFIG
    )
    classes = list(datagen.class_indices.keys())
    steps_per_epoch = len(datagen) // BATCH_SIZE
    validation_steps = len(datagen_val) // BATCH_SIZE
    test_steps = len(datagen_test) // BATCH_SIZE

    ds = create_classifier_dataset(datagen, IMAGE_SHAPE, len(classes))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(PREFETCH)

    ds_val = create_classifier_dataset(datagen_val, IMAGE_SHAPE, len(classes))
    ds_val = ds_val.batch(BATCH_SIZE)
    ds_val = ds_val.prefetch(PREFETCH)

    ds_test = create_classifier_dataset(datagen_test, IMAGE_SHAPE, len(classes))
    ds_test = ds_test.batch(BATCH_SIZE)
    ds_test = ds_test.prefetch(PREFETCH)

    return ds, ds_val, ds_test, (steps_per_epoch, validation_steps, test_steps), classes


def load_model(model_type, num_classes, steps_per_epoch, cifar_resnet,
               image_shape=(224, 224, 3), lr=5e-3, epochs=30,
               projector_dim=2048, evaluation=False,
               gpu_used=('GPU:0', 'GPU:1', 'GPU:2', 'GPU:3')):
    
    # DEBUG
    print(model_type, num_classes, steps_per_epoch, cifar_resnet, image_shape, lr, epochs, projector_dim, evaluation)
    
    strategy = tf.distribute.MirroredStrategy(gpu_used)
    print('Number of devices:', strategy.num_replicas_in_sync)

    with strategy.scope():
        if model_type not in ['supervised', 'barlow', 'barlow_fine_tuned']:
            raise ValueError
        encoder_trainable = False if ('barlow' in model_type and 'fine_tuned' not in model_type) else True
        print('Encoder trainable:', encoder_trainable)

        if evaluation:
            # Test time ==> no need to load pre-trained weights
            encoder_weights_path = None
        elif 'barlow' in model_type:
            model_name = 'encoder.h5' if cifar_resnet else 'resnet.h5'
            encoder_weights_path = os.path.join(PRETRAINED_DIR, model_name)
            print('Loading encoder weights from:', encoder_weights_path)
        else:
            encoder_weights_path = None

        if cifar_resnet:
            # Smaller modified ResNet20 that outputs a 256-d features?
            model = resnet_cifar.get_classifier(
                projector_dim=projector_dim,
                num_classes=num_classes,
                encoder_weights=encoder_weights_path,
                image_shape=image_shape
            )
        else:
            # Updated (larger) version of the encoder (ResNet50v2)
            model = resnet.get_classifier(
                num_classes=num_classes,
                input_shape=image_shape,
                encoder_weights=encoder_weights_path,
                encoder_trainable=encoder_trainable
            )

        # Set up optimizer
        warmup_epochs = 0.1
        warmup_steps = int(warmup_epochs * steps_per_epoch)

        lr_decay_fn = lr_scheduler.WarmUpCosine(
            learning_rate_base=lr,
            total_steps=epochs * steps_per_epoch,
            warmup_learning_rate=0.0,
            warmup_steps=warmup_steps
        )

        optimizer = SGDW(learning_rate=lr_decay_fn, momentum=0.9, nesterov=False, weight_decay=1e-6)
        
        model.summary()
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'acc',
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
                MatthewsCorrelationCoefficient(num_classes=num_classes, name='MCC')
            ]
        )

    return model


def main(suffix=None, model_name=None, cifar_resnet=True):
    save_dir = configure_saving(suffix, model_name)
    print('Saving at:', save_dir)

    ds, ds_val, ds_test, steps, classes = load_datasets()
    steps_per_epoch, validation_steps, test_steps = steps

    model = load_model(
        model_type=MODEL_TYPE, 
        num_classes=len(classes), 
        steps_per_epoch=steps_per_epoch, 
        cifar_resnet=cifar_resnet,
        image_shape=IMAGE_SHAPE,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        projector_dim=PROJECTOR_DIM,
        evaluation=False
    )

    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=PATIENCE)
    mc = ModelCheckpoint(
        os.path.join(save_dir, 'classifier.h5'),
        monitor='val_acc', mode='max',
        verbose=1,
        save_best_only=True, save_weights_only=True
    )

    print('\nSteps per epoch:', steps_per_epoch)
    history = model.fit(
        ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_data=ds_val,
        callbacks=[mc, es]
    )
    with open(os.path.join(save_dir, 'history.pickle'), 'wb') as file:
        pickle.dump(history.history, file)

    model.load_weights(os.path.join(save_dir, 'classifier.h5'))
    model.layers[1].save_weights(os.path.join(save_dir, 'encoder.h5'))

    model.evaluate(ds_test, steps=test_steps)


if __name__ == '__main__':
    MODEL_TYPE = ['barlow_fine_tuned', 'supervised'][0]
    PROJECTOR_DIM = 2048
    LEARNING_RATE = 5e-3

    
    PRETRAINED_DIR = f'trained_models/encoders/encoder_resnet50_100_baseline'
    ROOT_SAVE_DIR = 'trained_models/classifiers/resnet50_100_curve'
    
    DATASET_CONFIG['dataset_dir'] = 'datasets/tissue_classification/dataset_main_0.3'
    DATASET_CONFIG['split_file_path'] = 'datasets/tissue_classification/fold_test.csv'
    
    LEARNING_RATE = 0.5
    for s in [0.01]: 
        DATASET_CONFIG['train_split'] = s

        MODEL_TYPE = 'barlow_fine_tuned'
        main(model_name=f'barlow_{s}', cifar_resnet=False)
