from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
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

import yaml
import json
import os

from config.datasets_config import DATASETS_CONFIG


def load_datagens():
    datagens = []

    datagens.append(get_generators(
        splits=['test'],
        image_shape=config['image_shape'],
        batch_size=config['batch_size'],
        random_seed=config['random_seed'],
        dataset_config=dataset_config,
        separate_evaluation_groups=False
    )[0])
    datagens.extend(get_generators(
        splits=['test'],
        image_shape=config['image_shape'],
        batch_size=config['batch_size'],
        random_seed=config['random_seed'],
        dataset_config=dataset_config,
        separate_evaluation_groups=True
    )[0])

    return datagens, list(datagens[0].class_indices.keys())


def load_classifier():
    model = load_model(config=config, evaluation=True)
    model.load_weights(os.path.join(save_dir, 'classifier.h5'))

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
    print('Evaluating model from:', save_dir)

    # Create training curves, early stop values, etc.
    os.makedirs(visualization_save_dir, exist_ok=True)
    es_stats = analyze_history(
        os.path.join(save_dir, 'history.pickle'),
        save_visualization=True,
        return_es_stats=True,
        root_save_dir=visualization_save_dir
    )
    print('At the early stop epochs:', es_stats)

    # Evaluate classifier on test set
    dataset_config['train_split'] = config['train_split']
    dataset_config['validation_split'] = config['validation_split']
    datagens, classes = load_datagens()

    num_classes = len(classes)
    config['num_classes'] = num_classes
    config['steps_per_epoch'] = 1  # Used to set up optimizer

    model = load_classifier()

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
    with open('config/classifier_config.yaml') as file:
        config = yaml.safe_load(file)
        print(config)

    config['cifar_resnet'] = False
    np.random.seed(config['random_seed'])
    tf.random.set_seed(config['random_seed'])

    # For training on local machine
    config['dataset_type'] = 'tissue_5_0.5'
    dataset_config = DATASETS_CONFIG[config['dataset_type']]

    # Where the generated plots should be saved
    visualization_save_dir = 'visualization'

    # Where the model is saved
    save_dir = 'trained_models/classifiers/test_classifier'
    main()
