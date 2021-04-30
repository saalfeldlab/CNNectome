import os
import pkg_resources
import appdirs
import shutil
import configparser
import CNNectome


def get_config() -> configparser.ConfigParser:
    """
    Read configuration from system-dependent standard location. Will put in default version if no configuration file
    is found.

    Returns:
        Configuration object.
    """
    cfg_dir = appdirs.user_config_dir("CNNectome", version=CNNectome.__version__)
    cfg_file = os.path.join(cfg_dir, "config_local.ini")
    if not os.path.isfile(cfg_file):
        create_user_config(cfg_file)

    config = configparser.ConfigParser()
    config.read(cfg_file)
    return config


def create_user_config(cfg_file: str) -> None:
    """
    Copy example config file included in package data to `cfg_file`.

    Args:
        cfg_file: Destination path for configuration file.
    """
    source = pkg_resources.resource_filename("CNNectome", "etc/config_local.ini")
    os.makedirs(os.path.dirname(cfg_file), exist_ok=True)
    shutil.copyfile(source, cfg_file)
    print("Config file added here: {0:}".format(cfg_file))


if __name__ == "__main__":
    get_config()
