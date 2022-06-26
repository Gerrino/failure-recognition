from setuptools import setup, find_namespace_packages

required = [
    "sklearn",
    "click",
    "tqdm",
    "numpy<=1.22",
    "matplotlib",
    "pandas",
    "tsfresh",
]

setup(
    name="failure-recognition-signal-processing",
    packages=find_namespace_packages(include=["failure_recognition.*"]),
    install_requires=required,
    extras_require={
        "dev": ["pylint", "black"],
    },
)
