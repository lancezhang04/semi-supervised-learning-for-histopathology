from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
import tensorflow as tf
from utils import resnet20
from utils import lr_scheduler
import logging
import pickle
import os


# Doesn't seem to do anything for some TF versions
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)


# Configuration
TRAIN_FROM_SCRATCH = False  # Whether to train the model from scratch or use pretrained weights
PROJECTOR_DIMENSIONALITY = 2048
IMAGE_SHAPE = [64, 64, 3]
RANDOM_SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE = 25
EPOCHS = 100

PRETRAINED_DIR = 'resnet_encoders/best_model_2048_181.36.h5'
ROOT_DIR = 'datasets/NuCLS_64_7_0.2'

TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
VALIDATION_DIR = os.path.join(ROOT_DIR, 'val')
TEST_PATH = 'datasets/NuCLS_64_7/test'
CLASSES = os.listdir(TRAIN_DIR)
print(CLASSES)


# Load model
resnet_enc = resnet20.get_network(
    hidden_dim=PROJECTOR_DIMENSIONALITY,
    use_pred=False,
    return_before_head=False,
    input_shape=IMAGE_SHAPE
)
if not TRAIN_FROM_SCRATCH:
    resnet_enc.load_weights(PRETRAINED_DIR)
    # Freeze the weights
    resnet_enc.trainable = False

inputs = Input(IMAGE_SHAPE)
x = resnet_enc(inputs)
x = Dense(7, activation='softmax', kernel_initializer='he_normal')(x)
model = Model(inputs=inputs, outputs=x)


# Load data
# Notice that no data augmentation is used
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


# Train model
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=PATIENCE)
# this only save the best model
save_model_name = f'best_classifier.h5'
save_model_dir = os.path.join('trained_models', save_model_name)
mc = ModelCheckpoint(save_model_dir, monitor='val_acc', mode='max', verbose=1,
                     save_best_only=True, save_weights_only=True)

# Set up optimizer
STEPS_PER_EPOCH = len(datagen)
WARMUP_EPOCHS = int(EPOCHS * 0.1)
WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)

lr_decay_fn = lr_scheduler.WarmUpCosine(
    learning_rate_base=LEARNING_RATE,
    total_steps=EPOCHS * STEPS_PER_EPOCH,
    warmup_learning_rate=0.0,
    warmup_steps=WARMUP_STEPS
)

model.compile(
    optimizer=Adam(learning_rate=lr_decay_fn),
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
with open('benchmarks/history.pickle', 'wb') as file:
    pickle.dump(history.history, file)

model.load_weights(save_model_dir)
model.evaluate(datagen_test)

# [800, 547, 439, 349, 266, 216, 178, 152, 145, 136, 127, 119, 111, 101, 99, 97, 94, 90, 86, 81, 77, 73, 70, 68, 66, 64,62, 62, 60, 59, 58, 59, 57, 56, 55, 54, 53, 53, 52, 52, 52, 51, 51, 50, 50, 50, 50, 49, 49, 49, 49, 48, 48, 48, 48, 47, 47, 47, 47, 47]

"""
acc = metrics.CategoricalAccuracy()
   ...: acc.update_state(y_true, y_pred)
   ...: acc.result().numpy()

"""
