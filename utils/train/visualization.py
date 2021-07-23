import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def analyze_history(history_path, save_visualization=True, return_es_stats=True, root_save_dir='visualizations'):
    with open(history_path, 'rb') as file:
        history = pickle.load(file)

    val_acc = history['val_acc']
    early_stop_epoch = np.argmax(val_acc)

    if save_visualization:
        plt.figure(figsize=(10, 6))
        train_color, val_color = ['#c9590e', '#faac78'], ['#1860cc', '#8cb3ed']

        plt.plot(history['acc'], label='Training accuracy', c=train_color[0])
        plt.plot(val_acc, label='Validation accuracy', c=val_color[0])
        plt.plot(history['MCC'], label='Training MCC', c=train_color[1])
        plt.plot(history['val_MCC'], label='Validation MCC', c=val_color[1])

        # Graph a line at the early stop epoch
        plt.plot([early_stop_epoch, early_stop_epoch], [0, 1], 'k--', label='Early stop epoch')

        plt.legend()
        plt.savefig(os.path.join(root_save_dir, 'train_curves.png'))

    if return_es_stats:
        return {key: round(history[key][early_stop_epoch], 5) for key in history.keys()}


if __name__ == '__main__':
    analyze_history('../../trained_models/resnet_classifiers/1024/4/history.pickle')
