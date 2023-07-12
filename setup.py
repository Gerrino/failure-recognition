from setuptools import setup, find_namespace_packages
import sys

required = [
    "matplotlib",
    "pyyaml",
    "statsmodels",
    "scipy",
    "sklearn",
    "click",
    "tqdm",
    "numpy<=1.22",
    "pandas",
    "graphviz",
    "sympy",
    "pyod"
]

setup(
    name="failure-recognition",
    install_requires=required,
    extras_require={
        "dev": ["pylint", "black", "sphinx"],
    },    
)

