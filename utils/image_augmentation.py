import imgaug.augmenters as iaa
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def get_preprocessing_function(config, view):
    def img_func(images, random_state, parents, hooks):
        return [image.astype('uint8') for image in images]

    def scale_img_func(images, random_state, parents, hooks):
        return [(image.astype('float32') / 127.5 - 1) for image in images]

    def keypoint_func(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    aug = iaa.Sequential([
        # Change type to 'uint8'
        iaa.Lambda(img_func, keypoint_func),

        # Flipping
        iaa.Fliplr(0.5),
        iaa.Flipud(config['vertical_flip_probability']),

        # Color dropping
        iaa.Sometimes(config['color_dropping_probability'], iaa.Grayscale(alpha=1)),

        # Color jittering
        iaa.Sometimes(config['color_jittering'], iaa.Sequential([
            iaa.MultiplyBrightness((
                1 - config['brightness_adjustment_max_intensity'],
                1 + config['brightness_adjustment_max_intensity'])
            ),
            iaa.MultiplySaturation((
                1 - config['color_adjustment_max_intensity'],
                1 + config['color_adjustment_max_intensity'])
            ),
            iaa.MultiplyHue((
                1 - config['hue_adjustment_max_intensity'],
                1 + config['hue_adjustment_max_intensity'])
            ),
            iaa.GammaContrast((
                1 - config['contrast_adjustment_max_intensity'],
                1 + config['contrast_adjustment_max_intensity'])
            )
        ], random_order=True)),

        # Gaussian blurring and solarization
        iaa.Sometimes(config['gaussian_blurring_probability'][view], iaa.GaussianBlur(sigma=(0.1, 2))),
        iaa.Solarize(config['solarization_probability'][view]),

        iaa.Lambda(scale_img_func, keypoint_func)
    ])

    return aug


def augment(x: tf.Tensor, view, filter_size, config) -> tf.Tensor:
    # Input must be clipped into the range of [0, 1]

    # Random 90-degree rotations
    # x = tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    # Random horizontal and vertical flips
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_flip_left_right(x)

    if tf.random.uniform(shape=[]) < config['color_jittering']:
        funcs = [
            lambda: tf.image.random_hue(x, config['hue_adjustment_max_intensity']),
            lambda: tf.image.random_saturation(x, 1 - config['hue_adjustment_max_intensity'],
                                               1 + config['hue_adjustment_max_intensity']),
            lambda: tf.image.random_brightness(x, config['brightness_adjustment_max_intensity']),
            lambda: tf.image.random_contrast(x, 1 - config['contrast_adjustment_max_intensity'],
                                             1 + config['contrast_adjustment_max_intensity'])
        ]

        seq = tf.random.shuffle([i for i in range(len(funcs))])
        for i in seq:
            # Really weird implementation: cannot index with `i` directly since it is a Tensor
            for j in range(len(funcs)):
                if i == j:
                    x = funcs[j]()

    # Gaussian blur currently implemented through keras layer
    # if tf.random.uniform(shape=[]) < config['gaussian_blurring_probability'][view]:
    #     x = tfa.image.gaussian_filter2d(x, filter_shape=filter_size, sigma=1)

    if tf.random.uniform(shape=[]) < config['solarization_probability'][view]:
        x = tf.where(x < 0.5, x, 1 - x)

    return x


def get_gaussian_filter(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return h
