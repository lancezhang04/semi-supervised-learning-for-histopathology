import pickle
from optparse import OptionParser
import matplotlib.pyplot as plt
plt.style.use('dark_background')


parser = OptionParser()
parser.add_option('-p', '--path', dest='path', help='path to .pickle file', default='history.pickle')
(options, args) = parser.parse_args()


with open(options.path, 'rb') as file:
    history = pickle.load(file)
    print(history.keys())

max_val_acc = max(history['val_acc'])
max_val_epoch = history['val_acc'].index(max_val_acc)
print('Maximum validation accuracy: %.5f' % max_val_acc)
print('Test accuracy: %.5f' % history['test_acc'])

plt.figure(figsize=(10, 5))
plt.title('ResNet50 64x64 Images')

plt.plot(history['val_acc'], label='validation accuracy')
plt.plot(history['acc'], label='train accuracy')

plt.plot([max_val_epoch, max_val_epoch], [min(history['val_acc']), max(history['acc'])], '--r', alpha=0.3, label='early stop epoch')
plt.scatter([max_val_epoch], [max_val_acc], s=50, c='r', label='maximum validation accuracy: %.5f' % max_val_acc)
plt.scatter([max_val_epoch], [history['test_acc']], s=50, c='b', label='test accuracy: %.5f' % history['test_acc'])

plt.legend(loc='best')
plt.show()


plt.figure(figsize=(10, 5))
plt.title('ResNet50 64x64 Images')

plt.plot(history['val_loss'], label='validation loss')
plt.plot(history['loss'], label='train loss')

plt.plot([max_val_epoch, max_val_epoch], [min(history['loss']), max(history['val_loss'])], '--r', alpha=0.3, label='early stop epoch')

plt.legend(loc='best')
plt.show()


model_names = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
test_acc = [0.7248, 0.7296, 0.7357, 0.7200, 0.7313]

plt.figure(figsize=(10, 5))
plt.title('Accuracies of ResNet Models on 64x64 Dataset')
plt.bar(model_names, test_acc)
plt.ylim([min(test_acc) - 0.05, max(test_acc) + 0.05])
for i, v in enumerate(test_acc):
    plt.gca().text(i - 0.05, v + 0.003, str(v), color='white', fontweight='bold')
plt.show()
