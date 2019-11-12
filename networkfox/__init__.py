# Copyright 2019 Stanford University
# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

__author__ = 'hnguyen'
__version__ = '1.2.4'

from .functional import operation, compose, If, Else

# For backwards compatibility
from .base import Operation, Var
from .network import Network
