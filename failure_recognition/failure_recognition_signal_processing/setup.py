from setuptools import setup, find_namespace_packages

required = [
    "scikit-learn",
    "click",
    "tqdm",
    "numpy<=1.22",
    "matplotlib",
    "pandas",
    "cuda-python",
    "tsfresh",
    "sqlalchemy"
]

setup(
    name="failure-recognition-signal-processing",
    python_requires='<3.10',
    packages=find_namespace_packages(include=["failure_recognition.*"]),
    install_requires=required,
    extras_require={
        "dev": ["pylint", "black"],
    },
    include_package_data=True
)
