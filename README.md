# NetworkFoX

<!-- [![PyPI version](https://badge.fury.io/py/networkfox.svg)](https://badge.fury.io/py/networkfox) [![Build Status](https://travis-ci.org/yahoo/networkfox.svg?branch=master)](https://travis-ci.org/yahoo/networkfox) [![codecov](https://codecov.io/gh/yahoo/networkfox/branch/master/graph/badge.svg)](https://codecov.io/gh/yahoo/networkfox) -->

<!-- [Full Documentation](https://pythonhosted.org/networkfox/) -->

## Lightweight computation graphs for Python

NetworkFoX is a fork of graphkit, which adds the ability to do computations on top of NetworkX graphs. 
It stands for Network F of X. 

The features NetworkFoX adds to graphkit are:
- Control flow nodes, such as If, ElseIf, and Else.
- Type annotations and type checking for function inputs and outputs.
- Coloring nodes in a graph and evaluating only nodes of a given color.
- Picklable graphs.
- Fixing various bugs in graphkit.

## Quick start

Here's how to install:

```
pip install networkfox
```    

Here's a Python script with an example Networkfox computation graph that produces multiple outputs (`a * b`, `a - a * b`, and `abs(a - a * b) ** 3`):

```
from operator import mul, sub
from networkfox import compose, operation

# Computes |a|^p.
def abspow(a, p):
    c = abs(a) ** p
    return c

# Compose the mul, sub, and abspow operations into a computation graph.
graph = compose(name="graph")(
    operation(name="mul1", needs=["a", "b"], provides=["ab"])(mul),
    operation(name="sub1", needs=["a", "ab"], provides=["a_minus_ab"])(sub),
    operation(name="abspow1", needs=["a_minus_ab"], provides=["abs_a_minus_ab_cubed"], params={"p": 3})(abspow)
)

# Run the graph and request all of the outputs.
out = graph({'a': 2, 'b': 5})

# Prints "{'a': 2, 'a_minus_ab': -8, 'b': 5, 'ab': 10, 'abs_a_minus_ab_cubed': 512}".
print(out)

# Run the graph and request a subset of the outputs.
out = graph({'a': 2, 'b': 5}, outputs=["a_minus_ab"])

# Prints "{'a_minus_ab': -8}".
print(out)
```

As you can see, any function can be used as an operation in Networkfox, even ones imported from system modules!

# License

Code licensed under the Apache License, Version 2.0 license. See LICENSE file for terms.
