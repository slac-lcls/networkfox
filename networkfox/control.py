# Copyright 2018 Stanford University
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

"""
This sub-module contains statements that can be used for conditional evaluation of the graph.
"""

from .base import Control
from .functional import compose


class If(Control):

    def __init__(self, condition_needs, condition, **kwargs):
        super(If, self).__init__(**kwargs)
        self.condition_needs = condition_needs
        self.condition = condition
        self.order = 1

    def __call__(self, *args):
        self.graph = compose(name=self.name)(*args)
        return self

    def _compute_condition(self, named_inputs):
        inputs = [named_inputs[d] for d in self.condition_needs]
        return self.condition(*inputs)

    def _compute(self, named_inputs, color=None):
        return self.graph(named_inputs, color=color)


class Else(Control):

    def __init__(self, **kwargs):
        super(Else, self).__init__(**kwargs)
        self.order = 2

    def __call__(self, *args):
        self.graph = compose(name=self.name)(*args)
        return self

    def _compute(self, named_inputs, color=None):
        return self.graph(named_inputs, color=color)
