import yaml
import os


def log_config(config, ljust=50, save_config=False):
    """
    Prints all key-value pairs in a configuration dictionary and save it to a .yaml file
    """
    for k, v in config.items():
        print(k.ljust(ljust), v)
    print()

    if save_config:
        with open(os.path.join(config['save_dir'], 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
