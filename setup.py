#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To use: python setup.py install
#        python setup.py clean

import os

from setuptools import setup, Command
from setuptools.command.install import install


class CustomInstall(install):
    """
    Updates the given installer to create the __init__.py file and
    store the path to the package.
    """
    def run(self):
        install.run(self)


class CleanCommand(Command):
    
    """Custom clean command to tidy up the project root.
    From https://stackoverflow.com/questions/3779915"""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


setup(
    name='behalf',
    version='0.1.0',
    author='Ben Cook, Roxana Pop, Harshil Kamdar',
    author_email='bcook@cfa.harvard.edu',
    packages=['behalf'],
    url='https://github.com/bacook17/behalf',
    license='LICENSE',
    description="""BarnEs-Hut ALgorithm For CS205""",
    scripts=['behalf/run_behalf.py'],
    include_package_data=True,
    cmdclass={'clean': CleanCommand, 'install': CustomInstall},
    install_requires=[
        'mpi4py', 'numpy>=1.13.1',
    ],
    extras_require={"GPU": ['pycuda']},
)
