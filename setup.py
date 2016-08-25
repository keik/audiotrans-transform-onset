#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'scipy',
    'audiotrans'
]

setup(name='audiotrans-transform-onset',
      version='0.0.0',
      description="""audiotrans transform module for onset detection""",
      author='keik',
      author_email='k4t0.kei@gmail.com',
      url='https://github.com/keik/audiotrans-transform-onset',
      license='MIT',
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Developers',
          'Topic :: Multimedia :: Sound/Audio :: Conversion',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
      ],
      packages=find_packages(),
      install_requires=install_requires)
