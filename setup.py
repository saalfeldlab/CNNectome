import os
import sys
from setuptools import find_packages, setup

try:
    import z5py
except ModuleNotFoundError as e:
    raise type(e)(
        str(e)
        + " - 'z5py' dependency needs to be installed manually, it is not installable via "
        "pip"
    ).with_traceback(sys.exc_info()[2])


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
    "cremi",
    "gunpowder",
    "fuse",
]

EXTRAS = {
    "synapse_postprocessing": [
        "luigi"
    ],  # also needs simpleference, which is not installable via pip
    "malis_loss": ["malis"],
}

DEPENDENCY_LINKS = [
    "git+https://github.com/neptunes5thmoon/gunpowder.git@dist_transform_py3#egg=gunpowder",
    "git+https://github.com/neptunes5thmoon/fuse.git@intensity_augment#egg=fuse",
    "git+https://github.com/neptunes5thmoon/malis.git@fix_setup#egg=malis",
    "git+https://github.com/cremi/cremi_python.git#egg=cremi",
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
