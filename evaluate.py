from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
from config.classifier_default_config import *
from train_classifier import load_model
from utils.train.visualization import analyze_history
from utils.models import resnet_cifar, resnet
from utils.datasets import get_generators
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
from seaborn import heatmap
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_datagens():
    datagens = []

    with open(os.path.join(SAVE_DIR, 'dataset_config.json'), 'r') as file:
        DATASET_CONFIG = json.load(file)

    datagens.append(get_generators(
        splits=['test'],
        image_shape=IMAGE_SHAPE,
        batch_size=BATCH_SIZE,
        random_seed=RANDOM_SEED,
        config=DATASET_CONFIG,
        separate_evaluation_groups=False
    )[0])
    datagens.extend(get_generators(
        splits=['test'],
        image_shape=IMAGE_SHAPE,
        batch_size=BATCH_SIZE,
        random_seed=RANDOM_SEED,
        config=DATASET_CONFIG,
        separate_evaluation_groups=True
    )[0])

    return datagens, list(datagens[0].class_indices.keys())


def load_classifier(num_classes):
    model = load_model(
        model_type='supervised',
        num_classes=num_classes,
        steps_per_epoch=1,  # Only used to configure lr scheduler
        cifar_resnet=cifar_resnet,
        image_shape=IMAGE_SHAPE,
        projector_dim=PROJECTOR_DIM,
        evaluation=True
    )
    model.load_weights(os.path.join(SAVE_DIR, 'classifier.h5'))

    return model


def create_conf_matrix(datagen, model, classes):
    print('\nGenerating confusion matrix...')
    all_labels = []
    all_preds = []

    for _ in tqdm(range(len(datagen)), ncols=120):
        x, labels = next(datagen)
        preds = model.predict(x)
        all_labels.extend(labels)
        all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    conf_matrix = confusion_matrix(np.argmax(all_labels, axis=-1), np.argmax(all_preds, axis=-1), normalize='true')
    conf_matrix = pd.DataFrame(conf_matrix, columns=classes, index=classes)
    # confusion_matrix.set_index(CLASSES)

    plt.figure(figsize=(10, 8))
    heatmap(conf_matrix, annot=True)
    plt.ylabel('True Classes')
    plt.xlabel('Predicted Classes')
    plt.savefig(os.path.join(visualization_save_dir, 'confusion_matrix.png'))

    # Calculate AUROC
    for average in ['micro', 'macro', 'weighted']:
        score = roc_auc_score(all_labels, all_preds, average=average)
        print(average.title(), 'AUROC:', round(score, 4))
        
    # Calculate f1 scores
#     for average in ['micro', 'macro', 'weighted']:
#         score = f1_score(all_labels, all_preds, average=average)
#         print(average.title(), 'F1 score:', round(score, 4))


def main():
    print('Evaluating model from:', SAVE_DIR)

    # Create training curves, early stop values, etc.
    os.makedirs(visualization_save_dir, exist_ok=True)
    es_stats = analyze_history(
        os.path.join(SAVE_DIR, 'history.pickle'),
        save_visualization=True,
        return_es_stats=True,
        root_save_dir=visualization_save_dir
    )
    print('At the early stop epochs:', es_stats)

    # Evaluate classifier on test set
    datagens, classes = load_datagens()
    num_classes = len(classes)
    model = load_classifier(num_classes)

    for n, datagen in zip(['all', 'minor', 'major'], datagens):
        print(n + ':')
        model.evaluate(datagen)

    # Create confusion matrix and calculate AUROC scores
    create_conf_matrix(
        datagen=datagens[0],
        model=model,
        classes=classes
    )


if __name__ == '__main__':
    PROJECTOR_DIM = 2048
    BATCH_SIZE = 256
    
    visualization_save_dir = 'visualization'
    cifar_resnet = False
    
#     SAVE_DIR = 'trained_models/classifiers/resnet50_100_lr/barlow_0.5'
#     main()
    SAVE_DIR = 'trained_models/classifiers/resnet50_100_lr/barlow_0.5'
    main()
    
#     for folder in os.listdir('trained_models/classifiers/resnet50_30_curve'):
#         SAVE_DIR = os.path.join('trained_models/classifiers/resnet50_30_curve', folder)
#         main()
