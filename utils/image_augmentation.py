"""
The Barlow Twins paper used the following data augmentation methods:
1. random cropping
2. resizing to 224 x 224
3. horizontal flipping
4. color jittering (brightness, contrast, saturation, hue; order is randomly selected for each batch)
5. converting to grayscale
6. Gaussian blur
7. solarization

See the BYOL paper for implementation details
Barlow Twins: https://arxiv.org/pdf/2103.03230.pdf
BYOL: https://arxiv.org/pdf/2006.07733.pdf


Additional modifications for this project:
1. added vertical flip
2. instead of changing the order of the color jittering functions every batch, I just randomized it for each instance,
   which is easier to implement
3. removed random cropping and resizing; the images are already 64x64, random cropping would probably do more harm than
   good


TO-DO:
1. figure out if images are normalized/standardized
2. figure out what maximum intensity for color jittering really means...
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from functools import partial
from random import random
import numpy as np
import colorsys
from PIL.ImageFilter import GaussianBlur
from PIL import ImageEnhance, Image

preprocessing_config = {
    'color_jittering': 0.8,
    'color_dropping_probability': 0.2,
    'brightness_adjustment_max_intensity': 0.4,
    'contrast_adjustment_max_intensity': 0.4,
    'color_adjustment_max_intensity': 0.2,
    'hue_adjustment_max_intensity': 0.1,
    'gaussian_blurring_probability': [1.0, 0.1],  # the implementation might be different from the original paper
    'solarization_probability': [0, 0.2]
}


def apply_brightness_shift(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_contrast_shift(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def apply_color_shift(image, factor):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)


def shift_hue(arr, h_shift):
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = h + h_shift
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b))
    return arr


def apply_hue_shift(image, factor):
    arr = np.asarray(image).astype('float32')
    new_img = Image.fromarray(shift_hue(arr, factor - 1).astype('uint8'))
    return new_img


def jitter_color(image, config, verbose=0,
                 brightness_factor=None, contrast_factor=None, color_factor=None, hue_factor=None):
    """
    Randomly jitter the color of the input image:
    * Brightness
    * Contrast
    * Saturation (color)
    * Hue adjustment
    The transformations are applied at random order

    :param config: configuration used
    :param brightness_factor: randomized if not specified
    :param contrast_factor: randomized if not specified
    :param color_factor: randomized if not specified
    :param image: input image, three-dimensional array
    :param verbose
    :return: the modified image
    """
    image = Image.fromarray(np.asarray(image, dtype='uint8'))

    if not brightness_factor:
        brightness_factor = 1 + (random() - 0.5) * 2 * config['brightness_adjustment_max_intensity']
    if not contrast_factor:
        contrast_factor = 1 + (random() - 0.5) * 2 * config['contrast_adjustment_max_intensity']
    if not color_factor:
        color_factor = 1 + (random() - 0.5) * 2 * config['color_adjustment_max_intensity']
    if not hue_factor:
        hue_factor = 1 + (random() - 0.5) * 2 * config['hue_adjustment_max_intensity']
    if verbose > 0:
        print('brightness shift:', brightness_factor)
        print('contrast shift:', contrast_factor)
        print('color shift:', color_factor)

    transformation_functions = {
        0: partial(apply_brightness_shift, factor=brightness_factor),
        1: partial(apply_contrast_shift, factor=contrast_factor),
        2: partial(apply_color_shift, factor=color_factor),
        3: partial(apply_hue_shift, factor=hue_factor)
    }
    functions_order = np.arange(len(transformation_functions))
    np.random.shuffle(functions_order)
    if verbose > 0:
        print('functions order:', functions_order)

    for idx in functions_order:
        image = transformation_functions[idx](image)

    return np.array(image, dtype='float32')


def get_preprocessing_function(config, view, verbose=0):
    """
    :param config: configuration used
    :param view: either 0 or 1; the two view have different augmentation probabilities
    :param verbose
    :return: a preprocessing function used by the generator
    """

    def preprocessing_function(image):
        # randomly jitters the color of the image
        if random() < config['color_jittering']:
            image = jitter_color(image, config=config, verbose=verbose)

        # randomly transforms image to grayscale
        if random() < config['color_dropping_probability']:
            if verbose > 0:
                print('dropped color')
            image = image[:, :, 0] * 0.2989 + image[:, :, 1] * 0.5870 + image[:, :, 2] * 0.1140
            # restore to three channels
            image = np.expand_dims(image, 0)
            image = np.repeat(image, 3, axis=0)
            image = image.transpose(1, 2, 0)

        # randomly Gaussian blur the image
        if random() < config['gaussian_blurring_probability'][view]:
            image = Image.fromarray(np.asarray(image, dtype='uint8'))
            # the `radius` is basically standard deviation: [0.1, 2]
            image = image.filter(GaussianBlur(radius=random() * 1.9 + 0.1))
            image = np.array(image, dtype='float32')

        # randomly solarize the image
        image = image / 255  # normalize the image to [0, 1]
        if random() < config['solarization_probability'][view]:
            # the color is inverted above a threshold
            if verbose > 0:
                print('solarized')
            solarize = np.vectorize(lambda x: x if x < 0.5 else (1 - x))
            image = solarize(image)

        image = image * 2 - 1  # make the image values range [-1, 1]
        if verbose > 0:
            print('=' * 20)
        return np.asarray(image, dtype='float32')

    return preprocessing_function


def get_generator(config, view, verbose=0, validation_split=None, vertical_flip=True):
    generator = ImageDataGenerator(
        horizontal_flip=True,  # 50% probability
        vertical_flip=vertical_flip,  # 50% probability
        preprocessing_function=get_preprocessing_function(config=config, view=view, verbose=verbose),
        validation_split=validation_split
    )
    return generator


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    RANDOM_SEED = 42
    image_size = 64

    gen_a = get_generator(preprocessing_config, view=0, verbose=0)
    datagen_a = gen_a.flow_from_directory('../NuCLS_64_5/test', seed=RANDOM_SEED, target_size=image_size, batch_size=10)
    gen_b = get_generator(preprocessing_config, view=1, verbose=0)
    datagen_b = gen_b.flow_from_directory('../NuCLS_64_5/test', seed=RANDOM_SEED, target_size=image_size, batch_size=10)

    ims_a, _ = datagen_a.next()
    ims_b, _ = datagen_b.next()

    for idx in range(len(ims_a)):
        fig, axs = plt.subplots(2)
        axs[0].imshow(ims_a[idx] / 2 + 0.5)
        axs[1].imshow(ims_b[idx] / 2 + 0.5)

        plt.show()
