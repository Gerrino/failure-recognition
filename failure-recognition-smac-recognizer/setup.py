from setuptools import setup, find_namespace_packages

setup(
    name='failure-recognition-smac-recognizer',
    packages=find_namespace_packages(include=['failure_recognition.*']),
    install_requires=['tsfresh', 'sklearn', 'click', 'tqdm', 'pandas'],
)
