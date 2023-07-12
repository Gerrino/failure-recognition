from setuptools import setup, find_namespace_packages
import sys
__version__ = '1.0.8'

required = [
    "matplotlib",    
]

setup(
    name="failure_recognition",
    version=__version__,
    packages=find_namespace_packages(include=['failure_recognition.*']),
    install_requires=required,
    extras_require={
        "dev": ["pylint", "black", "sphinx"],
    },
    py_modules=['failure_recognition'],    
)

