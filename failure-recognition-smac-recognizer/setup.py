from setuptools import setup, find_namespace_packages

setup(
    name='failure-recognition-smac-recognizer',
    packages=find_namespace_packages(include=['failure_recognition.*']),
<<<<<<< HEAD
    install_requires=['tsfresh', 'sklearn', 'click', 'tqdm', 'pandas'],
=======
    install_requires=['numpy<=1.22', 'tsfresh', 'sklearn', 'click', 'tqdm', 'smac'],
>>>>>>> Ubuntu/GerritN/Organize
)
