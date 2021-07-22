from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
from tensorflow.keras.layers import Input, Dense
from tensorflow_addons.optimizers import SGDW
from tensorflow.keras.models import Model
import tensorflow as tf

from utils.datasets import get_generators, create_classifier_dataset
from utils.models.resnet import get_classifier
from utils.train import lr_scheduler
from utils.models import resnet_cifar


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
                     f'{PROJECTOR_DIMENSIONALITY}_' + \
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


def load_model(model_type, num_classes, steps_per_epoch, cifar_resnet, gpu_used=['GPU:0', 'GPU:1', 'GPU:2', 'GPU:3']):
    strategy = tf.distribute.MirroredStrategy(gpu_used)
    print('Number of devices:', strategy.num_replicas_in_sync)

    with strategy.scope():
        if model_type not in ['supervised', 'barlow', 'barlow_fine_tuned']:
            raise ValueError
        encoder_trainable = True if 'fine_tuned' not in model_type else False
        encoder_weights_path = os.path.join(PRETRAINED_DIR, 'encoder.h5') if 'barlow' in model_type else None

        if cifar_resnet:
            resnet_enc = resnet_cifar.get_network(
                hidden_dim=PROJECTOR_DIMENSIONALITY,
                use_pred=False,
                return_before_head=False,
                input_shape=IMAGE_SHAPE
            )

            if 'barlow' in model_type:
                resnet_enc.load_weights(encoder_weights_path)
            resnet_enc.trainable = encoder_trainable

            inputs = Input(IMAGE_SHAPE)
            x = resnet_enc(inputs)
            x = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
            model = Model(inputs=inputs, outputs=x)
        else:
            # Updated (larger) version of the encoder (ResNet50v2)
            model = get_classifier(
                num_classes=num_classes,
                input_shape=IMAGE_SHAPE,
                pretrained_dir=encoder_weights_path,
                encoder_trainable=encoder_trainable
            )

        # Set up optimizer
        warmup_epochs = 0.1
        warmup_steps = int(warmup_epochs * steps_per_epoch)

        lr_decay_fn = lr_scheduler.WarmUpCosine(
            learning_rate_base=LEARNING_RATE,
            total_steps=EPOCHS * steps_per_epoch,
            warmup_learning_rate=0.0,
            warmup_steps=warmup_steps
        )

        optimizer = SGDW(learning_rate=lr_decay_fn, momentum=0.9, nesterov=False, weight_decay=1e-6)

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

    model = load_model(MODEL_TYPE, len(classes), steps_per_epoch, cifar_resnet=cifar_resnet)

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
    model.layers[1].save_weights(os.path.join(save_dir, 'resnet_enc.h5'))

    model.evaluate(ds_test, steps=test_steps)


if __name__ == '__main__':
    MODEL_TYPE = 'supervised'
    PRETRAINED_DIR = f'trained_models/encoders/dim/encoder_2048'
    PROJECTOR_DIMENSIONALITY = 2048
    LEARNING_RATE = 5e-3

    # ROOT_SAVE_DIR = 'trained_models/encoders'

    # DATASET_CONFIG['dataset_dir'] = 'datasets/tissue_classification/dataset_super'
    DATASET_CONFIG['split_file_path'] = 'datasets/tissue_classification/fold_test.csv'
    for s in [0.2]:
        DATASET_CONFIG['train_split'] = s
        main(model_name=f'supervised_resnet50_{s}', cifar_resnet=False)
