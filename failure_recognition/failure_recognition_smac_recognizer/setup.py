"""setup module for the smac-recognizer"""

from setuptools import setup, find_namespace_packages

setup(
    name="failure-recognition-smac-recognizer",
    packages=find_namespace_packages(include=["failure_recognition.*"]),
    install_requires=["numpy<=1.22", "tsfresh", "sklearn", "click", "tqdm", "smac", "failure-recognition-signal-processing"],
    extras_require={
        "dev": ["pylint", "black"],
    },
)
