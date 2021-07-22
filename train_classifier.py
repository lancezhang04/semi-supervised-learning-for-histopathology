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
from utils.models import resnet


import numpy as np
import pickle
import json
import os

from config.classifier_default_config import *
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_splits(use_cross_validation, split_files_folder):
    if use_cross_validation:
        split_file_paths = [n for n in os.listdir(split_files_folder) if n.split('.')[1] == 'csv']
        split_file_paths = [os.path.join(split_files_folder,  n) for n in split_file_paths]
        print('Using', len(split_file_paths), 'fold validation')
        print('\n'.join(split_file_paths) + '\n')
    else:
        split_file_paths = [DATASET_CONFIG['split_file_path']]
    
    return split_file_paths


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


def load_classifier(model_type, num_classes, steps_per_epoch, gpu_used=['GRU:0', 'GPU:1', 'GPU:2', 'GPU:3']):
    strategy = tf.distribute.MirroredStrategy(gpu_used)
    print('Number of devices:', strategy.num_replicas_in_sync)

    with strategy.scope():
        resnet_enc = resnet.get_network(
            hidden_dim=PROJECTOR_DIMENSIONALITY,
            use_pred=False,
            return_before_head=False,
            input_shape=IMAGE_SHAPE
        )

        inputs = Input(IMAGE_SHAPE)
        x = resnet_enc(inputs)
        x = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
        model = Model(inputs=inputs, outputs=x)
        
        # Load pretrained weight if necessary
        if model_type not in ['supervised', 'barlow', 'barlow_fine_tuned']:
            raise ValueError
        if 'barlow' in model_type:
            resnet_enc.load_weights(os.path.join(PRETRAINED_DIR, 'encoder.h5'))
            if 'fine_tuned' not in model_type:
                resnet_enc.trainable = False

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


def main(suffix=None, model_name=None):
    save_dir = configure_saving(suffix, model_name)
    print('Saving at:', save_dir)

    ds, ds_val, ds_test, steps, classes = load_datasets()
    steps_per_epoch, validation_steps, test_steps = steps
    model = load_classifier(MODEL_TYPE, len(classes), steps_per_epoch)

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
#     for i in [0.01, 0.05, 0.2, 0.5, 0.85]:
#         DATASET_CONFIG['train_split'] = i
#         main(model_name=f'supervised_{i}_baseline')
    
    use_cross_validation = True
    # Folder containing all cross validation splits
    split_files_folder = 'datasets/tissue_classification/splits'
    split_file_paths = load_splits(use_cross_validation, split_files_folder)

    MODEL_TYPE = 'barlow_fine_tuned'
    PRETRAINED_DIR = f'trained_models/encoders/baseline_30'
    ROOT_SAVE_DIR = 'trained_models/classifiers/barlow_30_models'
    PROJECTOR_DIMENSIONALITY = 2048
    
    for split_file_path in split_file_paths:
        for s in [0.01, 0.2, 0.85]:
            DATASET_CONFIG['train_split'] = s
            main(model_name=f'barlow_{s}_{os.path.basename(split_file_path).split(".")[0]}')
