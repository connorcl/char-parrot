from configparser import ConfigParser

def load_config():
    """Read config file from the current directory"""
    config = ConfigParser()
    config.read('model.ini')
    return config['Model Configuration']