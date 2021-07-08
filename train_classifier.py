from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from tensorflow_addons.optimizers import SGDW
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Model
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
import tensorflow as tf
from utils.models import resnet20
from utils.train import lr_scheduler
from utils.datasets import get_generators, create_classifier_dataset
from optparse import OptionParser
from datetime import datetime
import pickle
import json
import os


parser = OptionParser()
parser.add_option('-t', '--train_from_scratch', dest='train_from_scratch', default=False, action='store_true')
parser.add_option('-f', '--fine_tune', dest='fine_tune', default=False, action='store_true')
(options, args) = parser.parse_args()


# ==================================================================================================================== #
# Configuration - dataset
# ==================================================================================================================== #
# region

# cell classification
# DATASET_CONFIG = {
#     'split': 'datasets/NuCLS/train_test_splits/fold_1_test.csv',
#     'train_split': 0.5,
#     'validation_split': 0.15,
#     'dataset_dir': 'datasets/NuCLS_histogram_matching/NuCLS_histogram_matching_64',
#     'groups': {
#         'tumor': 'tumor',
#         'fibroblast': 'stromal',
#         'vascular_endothelium': 'vascular_endothelium',
#         'macrophage': 'stromal',
#         'lymphocyte': 'stils',
#         'plasma_cell': 'stils',
#         'apoptotic_body': 'apoptotic_body'
#     },
#     'major_groups': ['tumor', 'stils']
# }

# tissue classification
DATASET_CONFIG = {
    'split': 'tissue_classification/fold_test.csv',
    'train_split': 0.5,
    'validation_split': 0.15,
    'dataset_dir': 'tissue_classification/tissue_classification',
    'groups': {
        'tumor': 'tumor',
        'dcis': 'tumor',
        'angioinvasion': 'tumor',
        'stroma': 'stroma',
        'necrosis_or_debris': 'necrosis_or_debris',
        'lymphocytic_infiltrate': 'inflammatory_infiltrate',
        'plasma_cells': 'inflammatory_infiltrate',
        'other_immune_infiltrate': 'inflammatory_infiltrate',
        'blood': 'other',
        'blood_vessel': 'other',
        'exclude': 'other',
        'fat': 'other',
        'glandular_secretions': 'other',
        'lymphatics': 'other',
        'metaplasia_NOS': 'other',
        'normal_acinus_or_duct': 'other',
        'outside_roi': 'other',
        'skin_adnexa': 'other',
        'undetermined': 'other'
    },
    'major_groups': ['tumor', 'stroma', 'inflammatory_infiltrate']
}
# endregion


# ==================================================================================================================== #
# Configuration - training
# ==================================================================================================================== #
# region

TRAIN_FROM_SCRATCH = False  # options.train_from_scratch
FINE_TUNE = True  # options.fine_tune
PROJECTOR_DIMENSIONALITY = 4096
IMAGE_SHAPE = [224, 224, 3]
RANDOM_SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
PATIENCE = 100
EPOCHS = 30

PRETRAINED_DIR = 'trained_models/encoders/encoder_tissue_224_4096_256_5_0.001/encoder_4096.h5'

ROOT_SAVE_DIR = 'trained_models/classifiers/0708'
# endregion


# ==================================================================================================================== #
# Saving information
# ==================================================================================================================== #
# region

dataset_type = 'tissue' if 'tissue' in DATASET_CONFIG['dataset_dir'] else 'cell'
if TRAIN_FROM_SCRATCH:
    model_type = 'supervised'
elif FINE_TUNE:
    model_type = 'barlow_fine_tune'
else:
    model_type = 'barlow'
model_name = f'{model_type}_' + \
             f'{dataset_type}_{IMAGE_SHAPE[0]}_{DATASET_CONFIG["train_split"]}_' + \
             f'{PROJECTOR_DIMENSIONALITY}_' + \
             f'{BATCH_SIZE}_{EPOCHS}_{LEARNING_RATE}'
print('Model name:', model_name)

SAVE_DIR = os.path.join(ROOT_SAVE_DIR, model_name)

try:
    os.makedirs(SAVE_DIR, exist_ok=False)
except:
    input_ = input('save_dir already exists, continue? (Y/n)  >> ')
    if input_ != 'Y':
        raise ValueError
        
with open(os.path.join(SAVE_DIR, 'dataset_config.json'), 'w') as file:
    json.dump(DATASET_CONFIG, file, indent=4)
# endregion


# ==================================================================================================================== #
# Load data generators
# ==================================================================================================================== #
# region

datagen, datagen_val, datagen_test = get_generators(
    ['train', 'val', 'test'], 
    IMAGE_SHAPE, 1,
    RANDOM_SEED, config=DATASET_CONFIG
)
ds = create_classifier_dataset(datagen)
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(40)

ds_val = create_classifier_dataset(datagen_val)
ds_val = ds_val.batch(BATCH_SIZE)
ds_val = ds_val.prefetch(40)

ds_test = create_classifier_dataset(datagen_test)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(40)

CLASSES = list(datagen.class_indices.keys())
STEPS_PER_EPOCH = len(datagen) // BATCH_SIZE
# endregion


# ==================================================================================================================== #
# Load model
# ==================================================================================================================== #
# region

strategy = tf.distribute.MirroredStrategy()
print('Number of devices:', strategy.num_replicas_in_sync)

with strategy.scope():
    resnet_enc = resnet20.get_network(
        hidden_dim=PROJECTOR_DIMENSIONALITY,
        use_pred=False,
        return_before_head=False,
        input_shape=IMAGE_SHAPE
    )
    if not TRAIN_FROM_SCRATCH:
        resnet_enc.load_weights(PRETRAINED_DIR)
        # Freeze the weights
        if not FINE_TUNE:
            resnet_enc.trainable = False

    inputs = Input(IMAGE_SHAPE)
    x = resnet_enc(inputs)
    x = Dense(len(CLASSES), activation='softmax', kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=x)
# endregion


# ==================================================================================================================== #
# Train model
# ==================================================================================================================== #
# region

# Save the dataset config
with open(os.path.join(SAVE_DIR, 'dataset_config.json'), 'w') as file:
    json.dump(DATASET_CONFIG, file)

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=PATIENCE)
mc = ModelCheckpoint(
    os.path.join(SAVE_DIR, 'classifier.h5'),
    monitor='val_acc', mode='max',
    verbose=1,
    save_best_only=True, save_weights_only=True
)
tensorboard_callback = TensorBoard(
    log_dir=os.path.join(SAVE_DIR, datetime.now().strftime('%Y%m%d-%H%M%S') + '.log'),
    update_freq='batch'
)

# Set up optimizer
WARMUP_EPOCHS = 0  # 10
WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)

lr_decay_fn = lr_scheduler.WarmUpCosine(
    learning_rate_base=LEARNING_RATE,
    total_steps=EPOCHS * STEPS_PER_EPOCH,
    warmup_learning_rate=0.0,
    warmup_steps=WARMUP_STEPS
)

# # Visualize the learning rate curve (why not)
# plt.plot(lr_decay_fn(tf.range(EPOCHS*STEPS_PER_EPOCH, dtype=tf.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")
# plt.show()

with strategy.scope():
    # SGD with momentum and weight decay of 1e-6
    optimizer = SGDW(learning_rate=lr_decay_fn, momentum=0.9, nesterov=False, weight_decay=1e-6)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'acc',
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
            MatthewsCorrelationCoefficient(num_classes=len(CLASSES), name='MCC')
        ]
    )

model.evaluate(ds_test)

# Save training history
history = model.fit(
    ds,
    epochs=EPOCHS,
    validation_data=ds_val,
    callbacks=[mc, es, tensorboard_callback]
)
with open(os.path.join(SAVE_DIR, 'history.pickle'), 'wb') as file:
    pickle.dump(history.history, file)


model.load_weights(os.path.join(SAVE_DIR, 'classifier.h5'))
model.layers[1].save_weights(os.path.join(SAVE_DIR, 'resnet_enc.h5'))
model.evaluate(ds_test)
# endregion
