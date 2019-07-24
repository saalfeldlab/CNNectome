from distutils.sysconfig import get_python_inc, get_config_var
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import os

include_dirs = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_scale_aff"),
    os.path.dirname(get_python_inc()),
    get_python_inc(),
]

library_dirs = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_scale_aff"),
    get_config_var("LIBDIR"),
]


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


setup(
    name="multi_scale_aff",
    version="0.2",
    description="Training and evaluation scripts for MALA (https://arxiv.org/abs/1709.02974).",
    url="https://github.com/funkey/mala",
    author="Jan Funke",
    author_email="jfunke@iri.upc.edu",
    license="MIT",
    cmdclass={"build_ext": build_ext},
    install_requires=["cython", "numpy"],
    setup_requires=["cython", "numpy"],
    packages=["multi_scale_aff"],
    ext_modules=[
        Extension(
            "multi_scale_aff.wrappers",
            ["multi_scale_aff/wrappers.pyx", "multi_scale_aff/multi_scale_aff.cxx"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            language="c++",
            extra_link_args=["-std=c++11"],
            extra_compile_args=["-std=c++11", "-w"],
        )
    ],
)
