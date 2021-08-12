# semi-supervised-learning-for-histopathology

## Encoder pre-training
| parameter name | default value | usage                                                                                                                                                                                                                                 |
|----------------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `batch_size`   | 256           | Batch size used during pre-training. According to the original Barlow Twins paper, model performance improves considerably with bigger batch sizes. However, using 4 Tesla V100 GPUs, the largest batch size we have achieved is 256. |
| `lr_base`      | 1e-3          | The base learning rate used for encoder training; it determines the highest value the `WarmUpCosine` scheduler reaches.                                                                                                               |
| `weight_decay` | 1.5e-6        | Determines the weight decay value if using a LAMB optimizer.                                                                                                                                                                          |
| `cifar_resnet` | False         | If set to `True`, the Barlow Twins encoder will be a ResNet18 designed for the CIFAR dataset; otherwise, the encoder will be a ResNet50v2. They have about 9 million and 40 million parameters, respectively.                         |
