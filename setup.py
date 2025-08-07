# setup.py
from setuptools import setup, find_packages

setup(
    name='custom_noise',
    version='0.1.0',
    description='A package to generate custom noise signals with 1/f^alpha PSD.',
    author='Ron Hinchley',
    author_email='commuted@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
