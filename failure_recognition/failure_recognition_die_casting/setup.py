from setuptools import setup, find_namespace_packages
import sys
required = ['sklearn', 'click', 'tqdm', 'numpy', 'failure-recognition-signal-processing', 'pandas', 'graphviz', 'sympy']

setup(
    name='failure-recognition-die-casting',
    packages=find_namespace_packages(include=['failure_recognition.*']),    
    install_requires=required,
    extras_require = {
       'dev': ['pylint', 'black', 'sphinx'],     
   }
)
#winget install graphviz