def log_config(config, ljust=50):
    for k, v in config.items():
        print(k.ljust(ljust), v)
    print()
