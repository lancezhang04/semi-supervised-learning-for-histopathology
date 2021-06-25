import pickle
import matplotlib.pyplot as plt
import numpy as np


HISTORY_PATH = 'benchmarks/barlow_0.1_history.pickle'


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


print(val_acc[early_stop_epoch], val_top_2_accuracy[early_stop_epoch], val_MCC[early_stop_epoch], val_loss[early_stop_epoch])
print(acc[early_stop_epoch], top_2_accuracy[early_stop_epoch], MCC[early_stop_epoch], loss[early_stop_epoch])
