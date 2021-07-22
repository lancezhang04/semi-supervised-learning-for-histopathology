from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import os


def get_encoder(input_shape=(224, 224, 3), weights=None):
    # Returns a ResNet50 encoder without the classification head
    encoder = ResNet50V2(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling='max'
    )
    return encoder


def get_classifier(num_classes, input_shape=(224, 224, 3), pretrained_dir=None, encoder_trainable=True):
    if pretrained_dir is not None:
        encoder_weights = os.path.join(pretrained_dir, 'encoder.h5')
    else:
        encoder_weights = None

    # Outputs a 2048-d feature
    encoder = get_encoder(input_shape=input_shape, weights=encoder_weights)
    encoder.trainable = encoder_trainable

    # Create classifier by adding one FC layer on top of encoder
    inputs = Input(input_shape)
    x = encoder(inputs)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=outputs)

    return model
