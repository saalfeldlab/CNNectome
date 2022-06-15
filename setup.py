import os
import sys
import warnings

from setuptools import find_packages, setup

NAME = "CNNectome"
DESCRIPTION = "A collection of scripts for building, training and validating Convolutional Neural Networks (CNNs) for Connectomics"
URL = "https://github.com/saalfeldlab/CNNectome"
EMAIL = "heinrichl@janelia.hhmi.org"
AUTHOR = "Larissa Heinrich"
REQUIRES_PYTHON = ">=3.6"

REQUIRED = [
    #    "tensorflow_gpu<1.15",
    "absl-py>=0.9",
    "appdirs",
    "dnspython",
    "numpy",
    "scipy<1.6",
    "cython",
    "h5py",
    "zarr>=2.4.0",
    "joblib",
    "lazy-property",
    "scikit-image",
    "matplotlib",
    "memory_profiler",
    "more-itertools",
    "pymongo",
    "scikit-learn",
    "SimpleITK",
    "tabulate",
    "corditea @ git+https://github.com/saalfeldlab/corditea",
    "cremi @ git+https://github.com/cremi/cremi_python@python3",
    "gunpowder @ git+https://github.com/neptunes5thmoon/gunpowder@dist_transform_py3",
    "fuse @ git+https://github.com/neptunes5thmoon/fuse@my_pipinstallable_version",
    "neptunes5thmoon-simpleference @ git+https://github.com/neptunes5thmoon/simpleference@master",
]

EXTRAS = {
    "synapse_postprocessing": ["luigi"],
    "malis_loss": ["malis @ git+https://github.com/neptunes5thmoon/malis@fix_setup"],
    "napari": ["napari"],
    "dev": ["pytest", "jupyter", "black"],
    "tf": "tensroflow_gpu<1.15",
}

DEPENDENCY_LINKS = [
    "git+https://github.com/saalfeldlab/corditea@main#egg=corditea",
    "git+https://github.com/cremi/cremi_python.git@python3#egg=cremi",
    "git+https://github.com/neptunes5thmoon/gunpowder.git@dist_transform_py3#egg=gunpowder",
    "git+https://github.com/neptunes5thmoon/fuse.git@my_pipinstallable_version#egg=fuse",
    "git+https://github.com/neptunes5thmoon/malis.git@fix_setup#egg=malis",
    "git+https://github.com/neptunes5thmoon/simpleference.git@master#egg=simpleference[zarr]"
    "git+https://github.com/neptunes5thmoon/simpleference.git@master#egg=neptunes5thmoon-simpleference",
]

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), "r") as f:
    LONG_DESCRIPTION = "\n" + f.read()
with open(os.path.join(here, "CNNectome", "VERSION"), "r") as version_file:
    VERSION = version_file.read().strip()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "add_missing_n5_attributes = CNNectome.utils.add_missing_n5_attributes:main",
            "auto_evaluation = CNNectome.validation.organelles.auto_evaluation:main",
            "init_CNNectome_config = CNNectome.utils.config_loader:get_config",
            "check_inference_complete = CNNectome.inference.check_inference_complete:main",
            "unet_inference = CNNectome.inference.unet_inference:main",
        ],
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    dependency_links=DEPENDENCY_LINKS,
    package_data={"CNNectome": ["etc/config_local.ini", "VERSION"]},
    include_package_data=True,
    license="BSD-2-Clause",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
