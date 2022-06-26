from setuptools import setup, find_namespace_packages

required = ['sklearn', 'click', 'tqdm', 'numpy']

setup(
    name='failure-recognition-die-casting',
    packages=find_namespace_packages(include=['failure_recognition.*']),
    install_requires=required,
    extras_require = {
       'dev': ['pylint', 'black', 'sphinx'],     
   }
)
