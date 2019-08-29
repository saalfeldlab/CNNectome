import os
import sys
from setuptools import find_packages, setup
import warnings

try:
    import z5py
except ModuleNotFoundError as e:
    warnings.warn(
        str(e)
        + " - 'z5py' dependency needs to be installed manually, it is not installable via "
        "pip",
        category=UserWarning,
    )


NAME = "CNNectome"
DESCRIPTION = "A collection of scripts for building, training and validating Convolutional Neural Networks (CNNs) for Connectomics"
URL = "https://github.com/saalfeldlab/CNNectome"
EMAIL = "heinrichl@janelia.hhmi.org"
AUTHOR = "Larissa Heinrich"
REQUIRES_PYTHON = ">=3.6"
VERSION = "2.0.dev1"

REQUIRED = [
    "tensorflow_gpu<2",
    "numpy",
    "scipy",
    "cython",
    "h5py",
    "joblib",
    "scikit-image",
    "matplotlib",
    "gunpowder @ git+https://github.com/neptunes5thmoon/gunpowder@dist_transform_py3",
    "fuse @ git+https://github.com/neptunes5thmoon/fuse@my_pipinstallable_version",
]

EXTRAS = {
    "synapse_postprocessing": [
        "luigi"
    ],  # also needs simpleference, which is not installable via pip
    "malis_loss": ["malis @ git+https://github.com/neptunes5thmoon/malis@fix_setup"],
}

DEPENDENCY_LINKS = [
    "git+https://github.com/neptunes5thmoon/gunpowder.git@dist_transform_py3#egg=gunpowder",
    "git+https://github.com/neptunes5thmoon/fuse.git@my_pipinstallable_version#egg=fuse",
    "git+https://github.com/neptunes5thmoon/malis.git@fix_setup#egg=malis",
]

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), "r") as f:
    LONG_DESCRIPTION = "\n" + f.read()

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
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    dependency_links=DEPENDENCY_LINKS,
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