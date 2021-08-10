from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


class ResetBatchIndex(Callback):
    """
    Resets the batch indeces of data generators after training ends
    There is currently a bug where the generators advance for three/four more steps
    after training ends (cause unknown)
    """
    def __init__(self, datagens):
        super().__init__()
        self.datagens = datagens
        
    def on_train_end(self, logs=None):
        print('Current batch indeces:', [gen.batch_index for gen in self.datagens])
        print('Resetting to 0')
        
        datagens[0].batch_index = 0
        datagens[1].batch_index = 0
        

class Logger(Callback):
    """
    Simple way to monitor jupyter notebook progress when cell output is disconnected
    """
    def __init__(self):
        super().__init__()
        
    def on_batch_end(self, batch, logs=None):
        with open('logs.txt', 'a') as f:
            f.write(str(batch) + ' ')
            
    def on_epoch_end(self, epoch, logs=None):
        with open('logs.txt', 'a') as f:
            f.write('\n\n')


class VAECheckpoint(Callback):
    def __init__(self, model, model_save_dir, latent_dim, test_batch):
        super().__init__()
        self.save_dir = os.path.join(model_save_dir, 'visualizations')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.eps = tf.random.normal(shape=(10, latent_dim))
        self.model = model
        self.test_batch = test_batch
        
    def on_epoch_end(self, epoch, logs=None):
        plt.figure(figsize=(20, 40))
        plt.axis('off')
        
        plt.imshow(np.vstack([
            self.stack_imgs(self.test_batch),
            self.stack_imgs(self.model(self.test_batch)),
            self.stack_imgs(self.model.sample(self.eps)),
        ]))
        
        plt.savefig(os.path.join(self.save_dir, f'epoch_{epoch}.png'))
       
    @staticmethod
    def stack_imgs(imgs):
        return np.vstack([
            np.hstack(np.array(imgs[i * 5:(i + 1) * 5])) for i in range(2)
        ])


class EncoderCheckpoint(Callback):
    def __init__(self, resnet_enc, save_dir):
        super().__init__()
        self.save_dir = os.path.join(save_dir, 'encoder.h5')
        self.min_loss = 1e5
        self.resnet_enc = resnet_enc

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] < self.min_loss:
            self.min_loss = logs['loss']
            print('\nSaving model, new lowest loss:', self.min_loss)
            self.resnet_enc.save_weights(self.save_dir)
            print('Finished saving ==> doing great!')  # debug message haha


# Taken from here: https://www.jeremyjordan.me/nn-learning-rate/
class LRFinder(Callback):
    """
    A simple callback for finding the optimal learning rate range for your model + dataset.

    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5,
                                 max_lr=1e-2,
                                 steps_per_epoch=np.ceil(epoch_size/batch_size),
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])

            lr_finder.plot_loss()
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient.

    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    """

    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

    def clr(self):
        """Calculate the learning rate."""
        x = self.iteration / self.total_iterations
        return self.min_lr + (self.max_lr - self.min_lr) * x

    def on_train_begin(self, logs=None):
        """Initialize the learning rate to the minimum value at the start of training."""
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        """Record previous batch statistics and update the learning rate."""
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

    def plot_lr(self):
        """Helper function to quickly inspect the learning rate schedule."""
        plt.plot(self.history['iterations'], self.history['lr'])
        # plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')

        plt.savefig('finder_learning_rate.png')
        plt.show()

    def plot_loss(self):
        """Helper function to quickly observe the learning rate experiment results."""
        plt.plot(self.history['lr'], self.history['loss'])
        # plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')

        plt.savefig('finder_loss.png')
        plt.show()
