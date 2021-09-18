import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import image_augmentation


def get_bt_datasets(df, config):
    datagen_a = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        df[df['split'] == 'train'],
        shuffle=False,
        seed=config['random_seed'],
        target_size=config['image_shape'][:2], 
        batch_size=config['batch_size']
    )

    datagen_b = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        df[df['split'] == 'train'],
        shuffle=False,
        seed=config['random_seed'],
        target_size=config['image_shape'][:2], 
        batch_size=config['batch_size']
    )
    
    ds_a = tf.data.Dataset.from_generator(lambda: [datagen_a.next()[0]], output_types='float32',
                                          output_shapes=[None] * 4)
    ds_a = ds_a.map(
        lambda x: image_augmentation.augment(x, 0, config=config['preprocessing_config']),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_a = ds_a.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds_b = tf.data.Dataset.from_generator(lambda: [datagen_b.next()[0]], output_types='float32',
                                          output_shapes=[None] * 4)
    ds_b = ds_b.map(
        lambda x: image_augmentation.augment(x, 1, config=config['preprocessing_config']),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_b = ds_b.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    steps = len(datagen_a)
    assert steps == len(datagen_b)
    config['steps'] = steps
                    
    return ds_a, ds_b
