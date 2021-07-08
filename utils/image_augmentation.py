import imgaug.augmenters as iaa
import tensorflow as tf
import tensorflow_addons as tfa


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


def augment(x: tf.Tensor, view, filter_size) -> tf.Tensor:
    # Gaussian blur's sigma is fixed at 1
    # Input must be clipped into the range of [0, 1]

    # Random 90-degree rotations
    x = tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # Random horizontal and vertical flips
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    if tf.random.uniform(shape=[]) < 0.8:
        funcs = [
            lambda: tf.image.random_hue(x, 0.1),
            lambda: tf.image.random_saturation(x, 0.8, 1.2),
            lambda: tf.image.random_brightness(x, 0.4),
            lambda: tf.image.random_contrast(x, 0.6, 1.4),
        ]

        seq = tf.random.shuffle([i for i in range(len(funcs))])
        for i in seq:
            # Really weird implementation: cannot index with `i` directly since it is a Tensor
            for j in range(len(funcs)):
                if i == j:
                    x = funcs[j]()

    if tf.random.uniform(shape=[]) < [1.0, 0.1][view]:
        x = tfa.image.gaussian_filter2d(x, filter_shape=filter_size, sigma=1)

    if tf.random.uniform(shape=[]) < [0, 0.2][view]:
        x = tf.where(x < 0.5, x, 1 - x)

    return x
