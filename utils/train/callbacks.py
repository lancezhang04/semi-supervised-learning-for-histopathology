from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


class ResetBatchIndex(Callback):
    """
    Resets the batch indeces of data generators after training ends
    There is currently a bug where the generator advances for three/four more steps
    after training ends (unknown cause)
    
    This is not game-breaking, but might be annoying when running experiments
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
            f.write(f'\nEpoch completed: {epoch}\n')


class VAECheckpoint(Callback):
    """
    Creates visualizations at the end of each VAE epoch
    Generates a image that contains the test_batch images, their reconstructions, and randomly generated images
    """
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
    """
    Saves the ResNet encoder of a BT model when a new lowest training loss is achieved
    """
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
