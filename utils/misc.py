import yaml
import os


def log_config(config, ljust=50, save_config=False):
    for k, v in config.items():
        print(k.ljust(ljust), v)
    print()

    if save_config:
        with open(os.path.join(config['save_dir'], 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
