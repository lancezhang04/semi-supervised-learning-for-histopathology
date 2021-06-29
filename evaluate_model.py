from sklearn.metrics import confusion_matrix, roc_auc_score
from utils.train.visualization import visualize_training
import matplotlib.pyplot as plt
from seaborn import heatmap
from optparse import OptionParser
import pandas as pd
from tqdm import tqdm
import numpy as np
import os


parser = OptionParser()
parser.add_option('-d', '--dir', dest='dir', default='trained_models/resnet_classifiers/1024/4')
parser.add_option('-n', '--name', dest='name', default='supervised_0.85')
parser.add_option('-v', '--no-visualization', dest='show_training_visualization', default=True, action='store_false')
parser.add_option('-s', '--no-stats', dest='calculate_stats', default=True, action='store_false')
(options, args) = parser.parse_args()


DIR = options.dir
NAME = options.name
TRAINABLE = False if ('barlow' in NAME and 'fine_tune' not in NAME) else True

SHOW_TRAINING_VISUALIZATION = options.show_training_visualization
CALCULATE_STATS = options.calculate_stats

IMAGE_SHAPE = [64, 64, 3]
PROJECTOR_DIMENSIONALITY = 1024
RANDOM_SEED = 42
BATCH_SIZE = 16
TEST_DIR = 'datasets/NuCLS_64_7_grouped/test'
CLASSES = os.listdir(TEST_DIR)

MODEL_NAME = NAME + '.h5'
HISTORY_NAME = NAME + '_history.pickle'
MODEL_PATH = os.path.join(DIR, MODEL_NAME)
HISTORY_PATH = os.path.join(DIR, HISTORY_NAME)

if SHOW_TRAINING_VISUALIZATION:
    visualize_training(HISTORY_PATH)

if CALCULATE_STATS:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
    from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
    from utils.models import resnet20
    import tensorflow as tf


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
    print('Micro AUROC:', round(micro_auroc, 4), 'Macro AUROC:', round(macro_auroc, 4))
