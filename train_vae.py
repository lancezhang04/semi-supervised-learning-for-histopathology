import yaml

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from utils.models.vae import CVAE
from utils.datasets import get_dataset_df


def load_dataset():
    df = get_dataset_df(config['dataset_config'], config['random_seed'], mode='encoder')
    df = df.sample(frac=1).reset_index(drop=True)

    datagen = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        df[df['split'] == 'train'],
        shuffle=False,
        seed=config['random_seed'],
        target_size=config['image_shape'][:2],
        batch_size=config['batch_size']
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: [datagen.next()[0]],
        output_types='float32', output_shapes=[None] * 4
    )
    dataset = dataset.map(lambda x: tf.clip_by_value(x, 0, 1), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(config['prefetch'])

    # Wrapper
    def get_generator():
        while True:
            yield next(iter(dataset))

    steps_per_epoch = len(datagen)
    config['steps_per_epoch'] = steps_per_epoch
    print('Steps per epoch:', steps_per_epoch)

    return get_generator()


def main(model_name=None):
    dataset = load_dataset()

    strategy = tf.distribute.MirroredStrategy(config['gpu_used'])
    print('Number of devices:', strategy.num_replicas_in_sync)

    with strategy.scope():
        # Convolutional variational autoencoder
        model = CVAE(
            latent_dim=config['latent_dim'],
            image_shape=config['image_shape']
        )


if __name__ == '__main__':
    with open('config/vae_config.yaml') as file:
        config = yaml.safe_load(file)
        print(config)

    main()
