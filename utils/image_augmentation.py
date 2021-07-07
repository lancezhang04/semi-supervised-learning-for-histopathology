import imgaug.augmenters as iaa


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

