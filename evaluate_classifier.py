from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score
from utils.train.visualization import visualize_training
from utils.datasets import get_generators
import matplotlib.pyplot as plt
from seaborn import heatmap
from optparse import OptionParser
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import os

parser = OptionParser()
parser.add_option('-d', '--dir', dest='dir', default='trained_models/classifiers/0707/supervised_0.5')
parser.add_option('-v', '--no-visualization', dest='show_training_visualization', default=True, action='store_false')
parser.add_option('-s', '--no-stats', dest='calculate_stats', default=True, action='store_false')
parser.add_option('--not-trainable', dest='trainable', default=True, action='store_false')
parser.add_option('-p', dest='projector_dim', default=1024)
(options, args) = parser.parse_args()


# ==================================================================================================================== #
# Configuration
# ==================================================================================================================== #
# region

DIR = options.dir
NAME = DIR.split('/')[-1]
TRAINABLE = options.trainable
OVERRIDE_DATASET_DIR = None  # 'datasets/NuCLS_histogram_matching/NuCLS_histogram_matching_64'

SHOW_TRAINING_VISUALIZATION = options.show_training_visualization
CALCULATE_STATS = options.calculate_stats

IMAGE_SHAPE = [224, 224, 3]
PROJECTOR_DIMENSIONALITY = options.projector_dim
RANDOM_SEED = 42
BATCH_SIZE = 128

tf.random.set_seed(RANDOM_SEED)
# endregion


# ==================================================================================================================== #
# Training visualization
# ==================================================================================================================== #
# region

if SHOW_TRAINING_VISUALIZATION:
    visualize_training(os.path.join(DIR, 'history.pickle'))
# endregion


# ==================================================================================================================== #
# Statistics
# ==================================================================================================================== #
# region

if CALCULATE_STATS:
    print('\nCalculating statistics...')
    
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
    from utils.models import resnet
    import tensorflow as tf

    # Load data
    with open(os.path.join(DIR, 'dataset_config.json'), 'r') as file:
        DATASET_CONFIG = json.load(file)
    if OVERRIDE_DATASET_DIR is not None:
        DATASET_CONFIG['dataset_dir'] = OVERRIDE_DATASET_DIR
    
    datagen_test = get_generators(['test'], IMAGE_SHAPE, BATCH_SIZE, RANDOM_SEED, DATASET_CONFIG)[0]
    datagen_test_minor, datagen_test_major = get_generators(
        ['test'],
        IMAGE_SHAPE, BATCH_SIZE,
        RANDOM_SEED, config=DATASET_CONFIG,
        separate_evaluation_groups=True
    )[0]
    CLASSES = list(datagen_test.class_indices.keys())

    # Load model
    resnet_enc = resnet.get_network(
        hidden_dim=PROJECTOR_DIMENSIONALITY,
        use_pred=False,
        return_before_head=False,
        input_shape=IMAGE_SHAPE
    )

    inputs = Input(IMAGE_SHAPE)
    x = resnet_enc(inputs)
    resnet_enc.trainable = TRAINABLE
    x = Dense(len(CLASSES), activation='softmax', kernel_initializer='he_normal')(x)
    
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
    model.load_weights(os.path.join(DIR, 'classifier.h5'))

    # Loss and metrics
    print('\nall:')
    test_loss, test_acc, test_top_2_acc, test_MCC = model.evaluate(datagen_test)
    print('minor:')
    test_loss_minor, test_acc_minor, test_top_2_acc_minor, test_MCC_minor = model.evaluate(datagen_test_minor)
    print('major:')
    test_loss_major, test_acc_major, test_top_2_acc_major, test_MCC_major = model.evaluate(datagen_test_major)

    # Create confusion matrix
    print('\ngenerating confusion matrix...')
    all_labels = []
    all_preds = []

    for idx in tqdm(range(len(datagen_test)), ncols=120):
        X, labels = next(datagen_test)
        preds = model.predict(X)
        all_labels.extend(labels)
        all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    conf_matrix = confusion_matrix(np.argmax(all_labels, axis=-1), np.argmax(all_preds, axis=-1), normalize='true')
    conf_matrix = pd.DataFrame(conf_matrix, columns=CLASSES, index=CLASSES)
    # confusion_matrix.set_index(CLASSES)

    plt.figure(figsize=(10, 8))
    heatmap(conf_matrix, annot=True)
    plt.ylabel('True Classes')
    plt.xlabel('Predicted Classes')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Calculate AUROC
    micro_auroc = roc_auc_score(all_labels, all_preds, average='micro')
    macro_auroc = roc_auc_score(all_labels, all_preds, average='macro')
    print('Micro AUROC:', round(micro_auroc, 4), 'Macro AUROC:', round(macro_auroc, 4))
# endregion
