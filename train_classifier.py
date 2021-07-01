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
from utils.datasets import get_generators
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
# =================================================================================================================== #
# Region

DATASET_CONFIG = {
    'split': 'datasets/NuCLS/train_test_splits/fold_1_test.csv',
    'train_split': 0.5,
    'validation_split': 0.15,
    'dataset_dir': 'datasets/NuCLS_128',
    'cell_groups': {
        'tumor': 'tumor',
        'fibroblast': 'stromal',
        'vascular_endothelium': 'vascular_endothelium',
        'macrophage': 'stromal',
        'lymphocyte': 'stils',
        'plasma_cell': 'stils',
        'apoptotic_body': 'apoptotic_body'
    },
    'major_groups': ['tumor', 'stils']
}
# Endregion


# ==================================================================================================================== #
# Configuration - training
# =================================================================================================================== #
# Region

TRAIN_FROM_SCRATCH = True  # options.train_from_scratch
FINE_TUNE = True  # options.fine_tune
PROJECTOR_DIMENSIONALITY = 1024
IMAGE_SHAPE = [64, 64, 3]
RANDOM_SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
PATIENCE = 100
EPOCHS = 30

PRETRAINED_DIR = 'trained_models/resnet_encoders/1024/encoder_1024_47.02.h5'

ROOT_SAVE_DIR = 'trained_models/resnet_classifiers/1024/5'
NAME = f'{"supervised" if TRAIN_FROM_SCRATCH else "barlow"}' + \
       f'{"_fine_tune" if not TRAIN_FROM_SCRATCH and FINE_TUNE else ""}_{DATASET_CONFIG["train_split"]}'

SAVE_DIR = os.path.join(ROOT_SAVE_DIR, NAME)
SAVED_MODEL_NAME = NAME + '.h5'
SAVED_HISTORY_NAME = NAME + '_history.pickle'

print('Saved model name:', SAVED_MODEL_NAME)
# Endregion


# ==================================================================================================================== #
# Load data generators
# ==================================================================================================================== #
# Region

datagen, datagen_val, datagen_test = get_generators(
    ['train', 'val', 'test'], IMAGE_SHAPE, BATCH_SIZE,
    DATASET_CONFIG, RANDOM_SEED
)
CLASSES = list(datagen.class_indices.keys())
# Endregion


# ==================================================================================================================== #
# Load model
# ==================================================================================================================== #
# Region

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
# Endregion


# ==================================================================================================================== #
# Train model
# ==================================================================================================================== #
# Region

os.mkdir(SAVE_DIR)

# Save the dataset config
with open(os.path.join(SAVE_DIR, 'dataset_config.json'), 'w') as file:
    json.dump(DATASET_CONFIG, file)

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=PATIENCE)
mc = ModelCheckpoint(
    os.path.join(SAVE_DIR, NAME + '.h5'),
    monitor='val_acc', mode='max',
    verbose=1,
    save_best_only=True, save_weights_only=True
)
tensorboard_callback = TensorBoard(
    log_dir=os.path.join(SAVE_DIR, datetime.now().strftime('%Y%m%d-%H%M%S') + '.log'),
    update_freq='batch'
)

# Set up optimizer
STEPS_PER_EPOCH = len(datagen)
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
# model.summary()

model.evaluate(datagen_test)

# Save training history
history = model.fit(
    datagen,
    epochs=EPOCHS,
    validation_data=datagen_val,
    callbacks=[mc, es, tensorboard_callback]
)
with open(os.path.join(SAVE_DIR, NAME + '_history.pickle'), 'wb') as file:
    pickle.dump(history.history, file)


model.load_weights(os.path.join(SAVE_DIR, NAME + '.h5'))
model.evaluate(datagen_test)
# Endregion
