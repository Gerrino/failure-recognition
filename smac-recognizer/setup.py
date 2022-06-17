from setuptools import setup, find_namespace_packages

setup(
    name='failure-recognition-smac-recognizer',
    packages=find_namespace_packages(include=['failure-recognition.*']),
    install_requires=['sklearn', 'click', 'tqdm'],
)
