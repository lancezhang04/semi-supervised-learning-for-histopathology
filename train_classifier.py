from silence_tensorflow import silence_tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons.optimizers import SGDW
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
import tensorflow as tf
from utils.models import resnet20
from utils.train import lr_scheduler
from optparse import OptionParser
import pickle
import os

silence_tensorflow()

parser = OptionParser()
parser.add_option('-d', '--dataset_dir', dest='dataset_dir', default='datasets/NuCLS_64_7_grouped_0.2')
parser.add_option('-t', '--train_from_scratch', dest='train_from_scratch', default=False, action='store_true')
parser.add_option('-f', '--fine_tune', dest='fine_tune', default=False, action='store_true')
(options, args) = parser.parse_args()


# ==================================================================================================================== #
# Configuration
# ==================================================================================================================== #
TRAIN_FROM_SCRATCH = options.train_from_scratch
FINE_TUNE = options.fine_tune  # Currently using constant learning rate
PROJECTOR_DIMENSIONALITY = 1024
IMAGE_SHAPE = [64, 64, 3]
RANDOM_SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
PATIENCE = 100
EPOCHS = 50  # 30

PRETRAINED_DIR = 'trained_models/resnet_encoders/1024/encoder_1024_47.02.h5'
DATASET_DIR = options.dataset_dir

SAVE_DIR = 'trained_models/resnet_classifiers/1024/4'
NAME = f'{"supervised" if TRAIN_FROM_SCRATCH else "barlow"}' + \
       f'{"_fine_tune" if not TRAIN_FROM_SCRATCH and FINE_TUNE else ""}_{DATASET_DIR.split("_")[-1]}'
SAVED_MODEL_NAME = NAME + '.h5'
SAVED_HISTORY_NAME = NAME + '_history.pickle'
print('Saved model name:', SAVED_MODEL_NAME)

TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VALIDATION_DIR = os.path.join(DATASET_DIR, 'val')
TEST_PATH = 'datasets/NuCLS_64_7_grouped/test'
CLASSES = os.listdir(TRAIN_DIR)
print(CLASSES)


# ==================================================================================================================== #
# Load data generators
# ==================================================================================================================== #
datagen = ImageDataGenerator()
datagen = datagen.flow_from_directory(
    TRAIN_DIR, seed=RANDOM_SEED, target_size=IMAGE_SHAPE[:2], batch_size=BATCH_SIZE
)

datagen_val = ImageDataGenerator()
datagen_val = datagen_val.flow_from_directory(
    VALIDATION_DIR, seed=RANDOM_SEED, target_size=IMAGE_SHAPE[:2], batch_size=BATCH_SIZE
)

datagen_test = ImageDataGenerator()
datagen_test = datagen_test.flow_from_directory(
    TEST_PATH, seed=RANDOM_SEED, target_size=IMAGE_SHAPE[:2], batch_size=BATCH_SIZE
)


# ==================================================================================================================== #
# Load model
# ==================================================================================================================== #
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


# ==================================================================================================================== #
# Train model
# ==================================================================================================================== #
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=PATIENCE)
# this only save the best model
save_model_dir = os.path.join(SAVE_DIR, SAVED_MODEL_NAME)

# mc = ModelCheckpoint(save_model_dir, monitor='val_loss', mode='min', verbose=1,
#                      save_best_only=True, save_weights_only=True)
mc = ModelCheckpoint(save_model_dir, monitor='val_acc', mode='max', verbose=1,
                     save_best_only=True, save_weights_only=True)

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
# optimizer = Adam(learning_rate=lr_decay_fn)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=[
        'acc',
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
        MatthewsCorrelationCoefficient(num_classes=len(CLASSES), name='MCC')
    ]
)
model.summary()

model.evaluate(datagen_test)

history = model.fit(datagen, epochs=EPOCHS, validation_data=datagen_val, callbacks=[mc, es])
with open(os.path.join(SAVE_DIR, SAVED_HISTORY_NAME), 'wb') as file:
    pickle.dump(history.history, file)


model.load_weights(save_model_dir)
model.evaluate(datagen_test)
