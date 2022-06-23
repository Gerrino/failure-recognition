from setuptools import setup, find_namespace_packages

setup(
    name='failure-recognition-die-casting',
    packages=find_namespace_packages(include=['failure_recognition.*']),
    install_requires=['sklearn', 'click', 'tqdm', 'scipy', 'numpy'],
)
