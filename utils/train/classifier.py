from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
from tensorflow_addons.optimizers import SGDW, MultiOptimizer
from tensorflow.keras.metrics import AUC, TopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from utils.train import lr_scheduler
from utils.models import resnet_cifar, resnet, vae


def get_optimizer(config_dict, base_lr):
    lr_fn = lr_scheduler.get_decay_fn(
        base_lr=base_lr,
        epochs=config_dict['epochs'],
        steps_per_epoch=config_dict['steps'][0]
    )
    
    if config_dict['optimizer'] == 'adam':
        return Adam(learning_rate=lr_fn)
    elif config_dict['optimizer'] == 'sgdw':
        return SGDW(learning_rate=lr_fn, momentum=0.9, weight_decay=0)
    else:
        raise ValueError


def load_model(config_dict):
    """Load and compile a classification model based on the given configuration"""
    strategy = tf.distribute.MirroredStrategy(config_dict['gpu_used'])
    print('Number of devices:', strategy.num_replicas_in_sync)

    with strategy.scope():
        # Build model (and load pretrained weights)
        model_build_functions = {
            'cifar': resnet_cifar.get_classifier,
            'resnet50': resnet.get_classifier,
            'vae': vae.get_classifier
        }
        model = model_build_functions[config_dict['model_type']](config_dict)

        # Set up optimizer and learning rate scheduler
        if config_dict['lr_scheduler'] == 'cosine':
            optimizers_and_layers = [
                (get_optimizer(config_dict, config_dict['encoder_lr']), model.layers[1]),  # encoder
                (get_optimizer(config_dict, config_dict['head_lr']), model.layers[2])      # classification head
            ]
            optimizer = MultiOptimizer(optimizers_and_layers)
        elif config_dict['lr_scheduler'] == 'plateau':
            # ReduceLROnPlateau does not support differential learning rates
            assert config_dict['encoder_lr'] == config_dict['head_lr']
            
            if config_dict['optimizer'] == 'adam':
                optimizer = Adam(config_dict['head_lr'])
            elif config_dict['optimizer'] == 'sgdw':
                optimizer = SGDW(learning_rate=config_dict['head_lr'], momentum=0.9, weight_decay=0)
            else:
                raise ValueError(f'"{config_dict["optimizer"]}" is not a supported optimizer type, please choose either "adam" or "sgdw"')
        else:
            raise ValueError(f'"{config_dict["lr_scheduler"]}" is not a supported scheduler type, please choose either "cosine" or "plateau"')
        
        # Print model summary and compile model
        print()
        model.summary()
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'acc',
                TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
                MatthewsCorrelationCoefficient(num_classes=config_dict['num_classes'], name='MCC'),
                AUC(multi_label=True, name='auc', num_thresholds=2000)
            ]
        )

    return model
