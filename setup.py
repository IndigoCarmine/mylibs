"""Minimal setup file for tasks project."""

from setuptools import setup, find_packages

setup(
    name='core',
    version='0.0.8',
    license='proprietary',
    description='Core Library for me',

    author='Indigo Carmine',
    author_email='tamanisikaminai@gmail.com',
    url='None.com',

    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    package_data={
        'core': ['py.typed',"*.pyi"],
    },
)

