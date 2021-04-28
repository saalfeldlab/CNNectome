import importlib.util
import os
import CNNectome.utils.label
import CNNectome.utils.cosem_db
from CNNectome.utils.crop_utils import check_label_in_crop
from gunpowder import Coordinate
from types import ModuleType
from typing import Dict, List
from CNNectome.utils import config_loader


def get_unet_setup(setup: str,
                   training_version: str = "v0003.2") -> ModuleType:
    """
    Load specific setup config.

    Args:
        setup: Setup to load.
        training_version: version of trainings associated with the desired setup script.

    Returns:
        Imported setup config as module.
    """
    setup_dirs = config_loader.get_config()["organelles"]["training_setups_paths"].split(",")
    for setup_dir_root in setup_dirs:
        setup_dir = os.path.join(setup_dir_root, training_version, setup)
        if os.path.exists(setup_dir):
            # setups can have names that are not python compatible (like include '.') so this trickery is necessary
            spec = importlib.util.spec_from_file_location("unet_template", os.path.join(setup_dir, "unet_template.py"))
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            return config
    raise FileNotFoundError("Setup config not found for {setup:}".format(setup=setup))


def detect_8nm(setup: str,
               training_version: str = "v0003.2") -> bool:
    """
    Check whether a setup is trained on 8nm data.

    Args:
        setup: Setup for which to check for training on 8nm data.
        training_version: version of trainings associated with the `setup`.

    Returns:

    """
    config = get_unet_setup(setup, training_version=training_version)
    if tuple(config.voxel_size_input) == (8, 8, 8):
        return True
    else:
        return False


def autodiscover_labels(setup: str,
                        training_version: str = "v0003.2") -> List[CNNectome.utils.label.Label]:
    """
    Load list of labels for a specific setup.

    Args:
        setup: Setup for which to load labels.
        training_version: version of trainings associated with the `setup`

    Returns:
        List of labels.
    """
    unet_setup = get_unet_setup(setup, training_version=training_version)
    return unet_setup.labels


def autodiscover_label_to_crops(setup: str,
                                db: CNNectome.utils.cosem_db.MongoCosemDB) -> Dict[str, List[str]]:
    """
    For each label trained by the setup get the list of validation crops that contain that label.

    Args:
        setup: Setup for which to get the label to crop dictionary.
        db: Database with crop information and evaluation results.

    Returns:
        Dictionary mapping labelnames to crop numbers.
    """
    labels = autodiscover_labels(setup, training_version=db.training_version)

    label_to_cropnos = {}
    crops = db.get_all_validation_crops()
    for lbl in labels:
        for crop in crops:
            if check_label_in_crop(lbl, crop):
                try:
                    label_to_cropnos[lbl.labelname].append(crop["number"])
                except KeyError:
                    label_to_cropnos[lbl.labelname] = [crop["number"]]
    if len(label_to_cropnos) < len(labels):
        setup_labels = set([lbl.labelname for lbl in labels])
        crop_labels = set(label_to_cropnos.keys())
        for lblname in setup_labels - crop_labels:
            print("{0:} not in any crop".format(lblname))
    return label_to_cropnos


def autodiscover_raw_datasets(setup: str,
                              training_version: str = "v0003.2") -> List[str]:
    """
    Associate the given setup with the list of possible raw datasets for prediction on the original datasets, depending
    on whether it's operating on 8nm or 4nm data.

    Args:
        setup: Setup for which to get the appropriate raw dataset.
        training_version: version of trainings associated with the `setup`.

    Returns:
        List of raw datasets.
    """
    is8 = detect_8nm(setup, training_version=training_version)
    if is8:
        raw_datasets = ["volumes/raw/s1", "volumes/subsampled/raw/0"]
    else:
        raw_datasets = ["volumes/raw"]
    return raw_datasets
