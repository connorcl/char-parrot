import os
import importlib

def load_config(path):
    """Load parameters from a config file"""
    config_path = '.'.join(os.path.splitext(os.path.normpath(path))[0].split(os.sep))
    config_dir = os.path.split(path)[0]
    if config_dir != '':
        # make sure there exists an __init__.py file in the directory
        # containing the config file if the config file does not reside
        # in the current directory, to allow importing it as a module
        init_file = os.path.join(config_dir, "__init__.py")
        if not os.path.isfile(init_file):
            f = open(init_file, 'w+')
            f.close()
    config = importlib.import_module(config_path)
    return config