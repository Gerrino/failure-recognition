from setuptools import setup, find_namespace_packages
import sys
__version__ = '1.0.8'

required = [
    "matplotlib",    
]

setup(
    name="failure_recognition",
    version=__version__,
    packages=find_namespace_packages(include=['failure_recognition.*', "failure_recognition_signal_proccessing.*", "failure_recognition.failure_recognition_signal_proccessing.*"]),
    install_requires=required,
    extras_require={
        "dev": ["pylint", "black", "sphinx"],
    },
    py_modules=['failure_recognition_signal_proccessing'],
    include_package_data=True
)

