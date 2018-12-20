#!/usr/bin/env python
# Copyright 2018 Stanford University
# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import os
import re
import io
from setuptools import setup

LONG_DESCRIPTION = """
Networkfox is a fork of graphkit that adds things like control flow, coloring, and picklable graphs.
"""

# Grab the version using convention described by flask
# https://github.com/pallets/flask/blob/master/setup.py#L10
with io.open('networkfox/__init__.py', 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

setup(
     name='networkfox',
     version=version,
     description='Lightweight computation graphs for Python',
     long_description=LONG_DESCRIPTION,
     author='Seshu Yamajala, Huy Nguyen, Arel Cordero, Pierre Garrigues, Rob Hess, Tobi Baumgartner, Clayton Mellina',
     author_email='',
     url='http://github.com/slac-lcls/networkfox',
     packages=['networkfox'],
     install_requires=['networkx'],
     extras_require={
          'plot': ['pydot', 'matplotlib']
     },
     tests_require=['numpy'],
     license='Apache-2.0',
     keywords=['graph', 'computation graph', 'DAG', 'directed acyclical graph'],
     classifiers=[
          'Development Status :: 5 - Production/Stable',
          'License :: OSI Approved :: Apache Software License',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development'
    ],
    platforms='Windows,Linux,Solaris,Mac OS-X,Unix'
)
