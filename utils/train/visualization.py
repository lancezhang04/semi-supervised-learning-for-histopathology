import matplotlib.pyplot as plt
import numpy as np
import pickle


def visualize_training(history_path):
    with open(history_path, 'rb') as file:
        history = pickle.load(file)

    val_acc = history['val_acc']
    early_stop_epoch = np.argmax(val_acc)

    plt.figure(figsize=(10, 6))
    # plt.style.use('dark_background')

    train_color, val_color = ['#c9590e', '#faac78'], ['#1860cc', '#8cb3ed']

    plt.plot(history['acc'], label='Training accuracy', c=train_color[0])
    plt.plot(val_acc, label='Validation accuracy', c=val_color[0])
    plt.plot(history['MCC'], label='Training MCC', c=train_color[1])
    plt.plot(history['val_MCC'], label='Validation MCC', c=val_color[1])
    # Graph a line at the early stop epoch
    plt.plot([early_stop_epoch, early_stop_epoch], [0, 1], 'k--', label='Early stop epoch')

    plt.legend()
    plt.show()

    print(f'At early stop epoch,')
    for key in history.keys():
        print(f'\t{key}: {round(history[key][early_stop_epoch], 5)}')


if __name__ == '__main__':
    visualize_training('../../trained_models/resnet_classifiers/1024/4/history.pickle')
