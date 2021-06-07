import os
import logging
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from utils.prev_model import ResNet50_with_classification_head
from optparse import OptionParser
from utils.models import ResnetBuilder
import pickle

# K.control_flow_ops.deprecation.silence()
# this silences tensorflow (for the most part)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)


# Example usage: python train.py --size 32 --batch_size 128 --model_name resnet50

parser = OptionParser()
parser.add_option('-s', '--size', dest='size', type='int')
parser.add_option('-b', '--batch_size', dest='batch_size', type='int')
parser.add_option('-l', '--learning_rate', dest='learning_rate', default=5e-5, type='float')
parser.add_option('-p', '--patience', dest='patience', default=10, type='int')
parser.add_option('-v', '--validation_split', dest='validation_split', default=0.15, type='float')
parser.add_option('-m', '--model_name', dest='model_name', default='resnet50')
(options, args) = parser.parse_args()

if not (options.size and options.batch_size):
    parser.error('Error: both image size and batch size must be provided')


# training configuration #
size = options.size
print('Using dataset with image size of', size)
batch_size = options.batch_size
patience = options.patience
learning_rate = options.learning_rate
validation_split = options.validation_split

model_name = options.model_name
print('Using', model_name)
model_functions = {
    'resnet152': ResnetBuilder.build_resnet_152,
    'resnet101': ResnetBuilder.build_resnet_101,
    'resnet50': ResnetBuilder.build_resnet_50,
    'resnet34': ResnetBuilder.build_resnet_34,
    'resnet18': ResnetBuilder.build_resnet_18,
    'prev_resnet50': ResNet50_with_classification_head  # previous implementation of resnet
}
# --------------------- #


train_path = f'NuCLS_{size}/train'
test_path = f'NuCLS_{size}/test'
save_dir = 'trained_models'

class_names = os.listdir(train_path)
class_names_test = os.listdir(test_path)
assert len(class_names) == len(class_names_test)
print(class_names)
print('# classes:', len(class_names))


# create training, validation, test generators
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=validation_split,
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

# remove later
# import matplotlib.pyplot as plt
# plt.hist(train_generator.classes)
# plt.show()
# plt.hist(validation_generator.classes)
# plt.show()
# plt.hist(test_generator.classes)
# plt.show()
# ---------- #

# create model
model = model_functions[model_name](
    input_shape=(size, size, 3),
    num_outputs=len(class_names)
)
model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['acc'])


# train model
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=patience)
# this only save the best model
save_model_name = f'{model_name}_{size}_{learning_rate}.h5'
mc = ModelCheckpoint(os.path.join(save_dir, save_model_name), monitor='val_acc', mode='max', verbose=1,
                     save_best_only=True, save_weights_only=True)

history = model.fit_generator(
    train_generator, validation_data=validation_generator,
    epochs=100, steps_per_epoch=len(train_generator), validation_steps=len(validation_generator),
    verbose=1, callbacks=[mc, es]
)


test_loss, test_acc = model.evaluate_generator(test_generator, steps=len(test_generator))
print(f'test_loss: {test_loss}\ntest_acc: {test_acc}')
history.history['test_loss'] = test_loss
history.history['test_acc'] = test_acc

save_history_name = f'{model_name}_{size}_{learning_rate}_{test_acc}_history.pickle'
with open(os.path.join(save_dir, save_history_name), 'wb') as f:
    pickle.dump(history.history, f)
print('Execution completed')
