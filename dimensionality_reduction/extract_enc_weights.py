from silence_tensorflow import silence_tensorflow
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from utils.models import resnet

silence_tensorflow()


IMAGE_SHAPE = (64, 64, 3)

resnet_enc = resnet.get_network(
    hidden_dim=1024,
    use_pred=False,
    return_before_head=False,
    input_shape=IMAGE_SHAPE
)

inputs = Input(IMAGE_SHAPE)
x = resnet_enc(inputs)
x = Dense(4, activation='softmax', kernel_initializer='he_normal')(x)
model = Model(inputs=inputs, outputs=x)

model.load_weights('trained_models/resnet_classifiers/1024/4/barlow_fine_tune_0.01.h5')

model.layers[1].save_weights('resnet_enc_barlow_fine_tune_0.01.h5')
