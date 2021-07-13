from tensorflow.keras.callbacks import Callback
import os


class EncoderCheckpoint(Callback):
    def __init__(self, resnet_enc, save_dir):
        super().__init__()
        self.save_dir = os.path.join(save_dir, 'encoder.h5')
        self.min_loss = 1e5
        self.resent_enc = resnet_enc

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] < self.min_loss:
            self.min_loss = logs['loss']
            print('\nSaving model, new lowest loss:', self.min_loss)
            self.resnet_enc.save_weights(self.save_dir)
