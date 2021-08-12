# semi-supervised-learning-for-histopathology

## Encoder pre-training
| parameter name | default value | usage                                                                                                                                                                                                         |
|----------------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `batch_size`   | 256           |                                                                                                                                                                                                               |
| `lr_base`      | 0.2           |                                                                                                                                                                                                               |
| `weight_decay` | 1.5e-6        |                                                                                                                                                                                                               |
| `cifar_resnet` | False         | If set to `True`, the Barlow Twins encoder will be a ResNet18 designed for the CIFAR dataset; otherwise, the encoder will be a ResNet50v2. They have about 9 million and 40 million parameters, respectively. |
