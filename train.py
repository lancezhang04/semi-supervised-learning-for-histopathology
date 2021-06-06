import os
import logging
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
# from utils.__models import ResNet50_with_classification_head
from utils.models import ResnetBuilder
import pickle

# followed article: https://machinelearningknowledge.ai/keras-implementation-of-resnet-50-architecture-from-scratch/
# K.control_flow_ops.deprecation.silence()
# this silences tensorflow (for the most part)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)


train_path = 'NuCLS_32/train'
test_path = 'NuCLS_32/test'
save_dir = 'trained_models'

size = 32
batch_size = 128
learning_rate = 5e-5
class_names = os.listdir(train_path)
class_names_test = os.listdir(test_path)
assert len(class_names) == len(class_names_test)

print(class_names)
print('# classes:', len(class_names))

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2,
    # zoom_range=0.1,
    # rotation_range=360,
    # zca_whitening=True,  # this messes everything up :)
    # brightness_range=[-0.2, 0.2]
)
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(train_path, target_size=(size, size), shuffle=True,
                                                    batch_size=batch_size, subset='training')
validation_generator = train_datagen.flow_from_directory(train_path, target_size=(size, size), shuffle=True,
                                                    batch_size=batch_size, subset='validation')
test_generator = test_datagen.flow_from_directory(test_path, target_size=(size, size), shuffle=True,
                                                  batch_size=batch_size)

# model = ResNet50_with_classification_head(
#     input_shape=(size, size, 3),
#     num_classes=len(class_names),
#     learning_rate=learning_rate
# )

model = ResnetBuilder.build_resnet_18(
    input_shape=(size, size, 3),
    num_outputs=len(class_names)
)
model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['acc'])
model.summary()


es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=20)
# this only save the best model
mc = ModelCheckpoint(f'model_{size}_{learning_rate}.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                     save_weights_only=True)
history = model.fit_generator(
    train_generator, validation_data=validation_generator,
    epochs=100, steps_per_epoch=len(train_generator), validation_steps=len(validation_generator),
    verbose=1, callbacks=[mc, es]
)


test_loss, test_acc = model.evaluate_generator(test_generator)
print(f'test_loss: {test_loss}\ntest_acc: {test_acc}')


with open('history.pickle', 'wb') as f:
    pickle.dump(history.history, f)
print('Execution completed')
