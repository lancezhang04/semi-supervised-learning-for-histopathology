from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from seaborn import heatmap
import pandas as pd
import tensorflow as tf
from utils.models import resnet20
from tqdm import tqdm
import pickle
import numpy as np
import os

NAME = 'supervised_0.85'
DIR = 'trained_models/resnet_classifiers/1024'

SHOW_TRAINING_VISUALIZATION = True

CALCULATE_STATS = False
IMAGE_SHAPE = [64, 64, 3]
PROJECTOR_DIMENSIONALITY = 1024
RANDOM_SEED = 42
BATCH_SIZE = 32
TEST_DIR = 'datasets/NuCLS_64_7/test'
CLASSES = os.listdir(TEST_DIR)

MODEL_NAME = NAME + '.h5'
HISTORY_NAME = NAME + '_history.pickle'
MODEL_PATH = os.path.join(DIR, MODEL_NAME)
HISTORY_PATH = os.path.join(DIR, HISTORY_NAME)

if SHOW_TRAINING_VISUALIZATION:
    with open(HISTORY_PATH, 'rb') as file:
        history = pickle.load(file)

    loss = history['loss']
    val_loss = history['val_loss']
    MCC = history['MCC']
    val_MCC = history['val_MCC']
    acc = history['acc']
    val_acc = history['val_acc']
    top_2_accuracy = history['top_2_accuracy']
    val_top_2_accuracy = history['val_top_2_accuracy']

    early_stop_epoch = np.argmax(val_acc)

    plt.figure(figsize=(10, 6))

    plt.plot(acc, label='Training accuracy')
    plt.plot(MCC, label='Training MCC')
    plt.plot(val_acc, label='Validation accuracy')
    plt.plot(val_MCC, label='Validation MCC')
    plt.plot([early_stop_epoch, early_stop_epoch], [0, 1], label='Early stop epoch')

    plt.legend()
    plt.show()

    print(val_acc[early_stop_epoch], val_top_2_accuracy[early_stop_epoch], val_MCC[early_stop_epoch],
          val_loss[early_stop_epoch])
    print(acc[early_stop_epoch], top_2_accuracy[early_stop_epoch], MCC[early_stop_epoch], loss[early_stop_epoch])

if CALCULATE_STATS:
    # Load data
    datagen_test = ImageDataGenerator()
    datagen_test = datagen_test.flow_from_directory(
        TEST_DIR, seed=RANDOM_SEED, target_size=IMAGE_SHAPE[:2], batch_size=BATCH_SIZE
    )

    # Load model
    resnet_enc = resnet20.get_network(
        hidden_dim=PROJECTOR_DIMENSIONALITY,
        use_pred=False,
        return_before_head=False,
        input_shape=IMAGE_SHAPE
    )

    inputs = Input(IMAGE_SHAPE)
    x = resnet_enc(inputs)
    if 'barlow' in MODEL_PATH:
        resnet_enc.trainable = False
    x = Dense(7, activation='softmax', kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'acc',
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
            MatthewsCorrelationCoefficient(num_classes=len(CLASSES), name='MCC')
        ]
    )
    model.load_weights(MODEL_PATH)

    # Loss and metrics
    test_loss, test_acc, test_top_2_acc, test_MCC = model.evaluate(datagen_test)

    # Create confusion matrix
    all_labels = []
    all_preds = []

    for idx in tqdm(range(len(datagen_test))):
        X, labels = next(datagen_test)
        preds = model.predict(X)
        all_labels.extend(labels)
        all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    conf_matrix = confusion_matrix(np.argmax(all_labels, axis=-1), np.argmax(all_preds, axis=-1)).astype('int32')
    conf_matrix = pd.DataFrame(conf_matrix, columns=CLASSES, index=CLASSES)
    # confusion_matrix.set_index(CLASSES)

    plt.figure(figsize=(10, 8))
    heatmap(conf_matrix, annot=True)
    plt.ylabel('True Classes')
    plt.xlabel('Predicted Classes')
    plt.show()

    # Calculate AUROC
    micro_auroc = roc_auc_score(all_labels, all_preds, average='micro')
    macro_auroc = roc_auc_score(all_labels, all_preds, average='macro')
    print('Micro AUROC:', micro_auroc, 'Macro AUROC:', macro_auroc)
