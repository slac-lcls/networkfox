# Copyright 2018 Stanford University
# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

import math

from pprint import pprint
from operator import add, sub, mul
from numpy.testing import assert_raises

import networkfox.modifiers as modifiers
from networkfox import operation, compose, If, ElseIf, Else, Var


def test_network():

    # Sum operation, late-bind compute function
    sum_op1 = operation(name='sum_op1', needs=['a', 'b'], provides='sum_ab')(add)

    # sum_op1 is callable
    print(sum_op1(1, 2))

    # Multiply operation, decorate in-place
    @operation(name='mul_op1', needs=['sum_ab', 'b'], provides='sum_ab_times_b')
    def mul_op1(a, b):
        return a * b

    # mul_op1 is callable
    print(mul_op1(1, 2))

    # Pow operation
    @operation(name='pow_op1', needs='sum_ab', provides=['sum_ab_p1', 'sum_ab_p2', 'sum_ab_p3'], params={'exponent': 3})
    def pow_op1(a, exponent=2):
        return [math.pow(a, y) for y in range(1, exponent+1)]

    print(pow_op1._compute({'sum_ab':2}, ['sum_ab_p2']))

    # Partial operation that is bound at a later time
    partial_op = operation(name='sum_op2', needs=['sum_ab_p1', 'sum_ab_p2'], provides='p1_plus_p2')

    # Bind the partial operation
    sum_op2 = partial_op(add)

    # Sum operation, early-bind compute function
    sum_op_factory = operation(add)

    sum_op3 = sum_op_factory(name='sum_op3', needs=['a', 'b'], provides='sum_ab2')

    # sum_op3 is callable
    print(sum_op3(5, 6))

    # compose network
    net = compose(name='my network')(sum_op1, mul_op1, pow_op1, sum_op2, sum_op3)

    #
    # Running the network
    #

    # get all outputs
    pprint(net({'a': 1, 'b': 2}))

    # get specific outputs
    pprint(net({'a': 1, 'b': 2}, outputs=["sum_ab_times_b"]))

    # start with inputs already computed
    pprint(net({"sum_ab": 1, "b": 2}, outputs=["sum_ab_times_b"]))

    # visualize network graph
    # net.plot(show=True)


def test_network_simple_merge():

    sum_op1 = operation(name='sum_op1', needs=['a', 'b'], provides='sum1')(add)
    sum_op2 = operation(name='sum_op2', needs=['a', 'b'], provides='sum2')(add)
    sum_op3 = operation(name='sum_op3', needs=['sum1', 'c'], provides='sum3')(add)
    net1 = compose(name='my network 1')(sum_op1, sum_op2, sum_op3)
    pprint(net1({'a': 1, 'b': 2, 'c': 4}))

    sum_op4 = operation(name='sum_op1', needs=['d', 'e'], provides='a')(add)
    sum_op5 = operation(name='sum_op2', needs=['a', 'f'], provides='b')(add)
    net2 = compose(name='my network 2')(sum_op4, sum_op5)
    pprint(net2({'d': 1, 'e': 2, 'f': 4}))

    net3 = compose(name='merged')(net1, net2)
    pprint(net3({'c': 5, 'd': 1, 'e': 2, 'f': 4}))


def test_network_deep_merge():

    sum_op1 = operation(name='sum_op1', needs=['a', 'b'], provides='sum1')(add)
    sum_op2 = operation(name='sum_op2', needs=['a', 'b'], provides='sum2')(add)
    sum_op3 = operation(name='sum_op3', needs=['sum1', 'c'], provides='sum3')(add)
    net1 = compose(name='my network 1')(sum_op1, sum_op2, sum_op3)
    pprint(net1({'a': 1, 'b': 2, 'c': 4}))

    sum_op4 = operation(name='sum_op1', needs=['a', 'b'], provides='sum1')(add)
    sum_op5 = operation(name='sum_op4', needs=['sum1', 'b'], provides='sum2')(add)
    net2 = compose(name='my network 2')(sum_op4, sum_op5)
    pprint(net2({'a': 1, 'b': 2}))

    net3 = compose(name='merged', merge=True)(net1, net2)
    pprint(net3({'a': 1, 'b': 2, 'c': 4}))


def test_input_based_pruning():
    # Tests to make sure we don't need to pass graph inputs if we're provided
    # with data further downstream in the graph as an input.

    sum1 = 2
    sum2 = 5

    # Set up a net such that if sum1 and sum2 are provided directly, we don't
    # need to provide a and b.
    sum_op1 = operation(name='sum_op1', needs=['a', 'b'], provides='sum1')(add)
    sum_op2 = operation(name='sum_op2', needs=['a', 'b'], provides='sum2')(add)
    sum_op3 = operation(name='sum_op3', needs=['sum1', 'sum2'], provides='sum3')(add)
    net = compose(name='test_net')(sum_op1, sum_op2, sum_op3)

    results = net({'sum1': sum1, 'sum2': sum2})

    # Make sure we got expected result without having to pass a or b.
    assert 'sum3' in results
    assert results['sum3'] == add(sum1, sum2)


def test_output_based_pruning():
    # Tests to make sure we don't need to pass graph inputs if they're not
    # needed to compute the requested outputs.

    c = 2
    d = 3

    # Set up a network such that we don't need to provide a or b if we only
    # request sum3 as output.
    sum_op1 = operation(name='sum_op1', needs=['a', 'b'], provides='sum1')(add)
    sum_op2 = operation(name='sum_op2', needs=['c', 'd'], provides='sum2')(add)
    sum_op3 = operation(name='sum_op3', needs=['c', 'sum2'], provides='sum3')(add)
    net = compose(name='test_net')(sum_op1, sum_op2, sum_op3)

    results = net({'c': c, 'd': d}, outputs=['sum3'])

    # Make sure we got expected result without having to pass a or b.
    assert 'sum3' in results
    assert results['sum3'] == add(c, add(c, d))


def test_input_output_based_pruning():
    # Tests to make sure we don't need to pass graph inputs if they're not
    # needed to compute the requested outputs or of we're provided with
    # inputs that are further downstream in the graph.

    c = 2
    sum2 = 5

    # Set up a network such that we don't need to provide a or b d if we only
    # request sum3 as output and if we provide sum2.
    sum_op1 = operation(name='sum_op1', needs=['a', 'b'], provides='sum1')(add)
    sum_op2 = operation(name='sum_op2', needs=['c', 'd'], provides='sum2')(add)
    sum_op3 = operation(name='sum_op3', needs=['c', 'sum2'], provides='sum3')(add)
    net = compose(name='test_net')(sum_op1, sum_op2, sum_op3)

    results = net({'c': c, 'sum2': sum2}, outputs=['sum3'])

    # Make sure we got expected result without having to pass a, b, or d.
    assert 'sum3' in results
    assert results['sum3'] == add(c, sum2)


def test_pruning_raises_for_bad_output():
    # Make sure we get a ValueError during the pruning step if we request an
    # output that doesn't exist.

    # Set up a network that doesn't have the output sum4, which we'll request
    # later.
    sum_op1 = operation(name='sum_op1', needs=['a', 'b'], provides='sum1')(add)
    sum_op2 = operation(name='sum_op2', needs=['c', 'd'], provides='sum2')(add)
    sum_op3 = operation(name='sum_op3', needs=['c', 'sum2'], provides='sum3')(add)
    net = compose(name='test_net')(sum_op1, sum_op2, sum_op3)

    # Request two outputs we can compute and one we can't compute.  Assert
    # that this raises a ValueError.
    assert_raises(ValueError, net, {'a': 1, 'b': 2, 'c': 3, 'd': 4},
                  outputs=['sum1', 'sum3', 'sum4'])


def test_optional():
    # Test that optional() needs work as expected.

    # Function to add two values plus an optional third value.
    def addplusplus(a, b, c=0):
        return a + b + c

    sum_op = operation(name='sum_op1', needs=['a', 'b', modifiers.optional('c')], provides='sum')(addplusplus)

    net = compose(name='test_net')(sum_op)

    # Make sure output with optional arg is as expected.
    named_inputs = {'a': 4, 'b': 3, 'c': 2}
    results = net(named_inputs)
    assert 'sum' in results
    assert results['sum'] == sum(named_inputs.values())

    # Make sure output without optional arg is as expected.
    named_inputs = {'a': 4, 'b': 3}
    results = net(named_inputs)
    assert 'sum' in results
    assert results['sum'] == sum(named_inputs.values())


def test_deleted_optional():
    # Test that DeleteInstructions included for optionals do not raise
    # exceptions when the corresponding input is not prodided.

    # Function to add two values plus an optional third value.
    def addplusplus(a, b, c=0):
        return a + b + c

    # Here, a DeleteInstruction will be inserted for the optional need 'c'.
    sum_op1 = operation(name='sum_op1', needs=['a', 'b', modifiers.optional('c')], provides='sum1')(addplusplus)
    sum_op2 = operation(name='sum_op2', needs=['sum1', 'sum1'], provides='sum2')(add)
    net = compose(name='test_net')(sum_op1, sum_op2)

    # DeleteInstructions are used only when a subset of outputs are requested.
    results = net({'a': 4, 'b': 3}, outputs=['sum2'])
    assert 'sum2' in results


def test_control():

    # create graph with control flow (if, elseif, else)
    graph = compose(name='graph')(
        operation(name="mul1", needs=['a', 'b'], provides=['ab'])(mul),
        If(name='if_less_than_2', needs=['ab'], provides=['d'], condition_needs=['i'], condition=lambda i: i < 2)(
            operation(name='add', needs=['ab'], provides=['c'])(lambda ab: ab + 2),
            operation(name='sub2', needs=['c'], provides=['d'])(lambda c: c - 2)
        ),
        ElseIf(name='elseif', needs=['ab'], provides=['d'], condition_needs=['ab'], condition=lambda ab: ab > 2)(
            operation(name='add', needs=['ab'], provides=['d'])(lambda ab: ab*10)
        ),
        Else(name='else_less_than_2', needs=['ab'], provides=['d'])(
            operation(name='sub', needs=['ab'], provides=['c'])(lambda ab: ab - 1),
            operation(name='add2', needs=['c'], provides=['d'])(lambda c: c + 1)
        ),
        operation(name='div', needs=['d'], provides=['e'])(lambda d: d/2)
    )

    # check if branch
    results = graph({'a': 1, 'b': 3, 'i': 1})
    assert results == {'ab': 3, 'c': 5, 'd': 3, 'e': 1.5}

    # check else if branch
    results = graph({'a': 1, 'b': 3, 'i': 3})
    assert results == {'ab': 3, 'd': 30, 'e': 15.0}

    # check else branch
    results = graph({'a': 1, 'b': 1, 'i': 3})
    assert results == {'ab': 1, 'c': 0, 'd': 1, 'e': 0.5}


def test_color():
    graph = compose(name='graph')(
        operation(name='sum', needs=['a', 'b'], provides=['apb'], color='red')(add),
        operation(name='mul', needs=['a', 'b'], provides=['ab'], color='blue')(mul)
    )

    res = graph({'a': 2, 'b': 3}, color='red')
    assert res == {'apb': 5}

    res = graph({'a': 2, 'b': 3}, color='blue')
    assert res == {'ab': 6}


def test_control_and_color():
    graph = compose(name='graph')(
        operation(name="mul1", needs=['a', 'b'], provides=['ab'], color='red')(mul),
        If(name='if_less_than_2', needs=['ab'], provides=['d'], condition_needs=['i'], condition=lambda i: i < 2)(
            operation(name='add', needs=['ab'], provides=['c'], color='red')(lambda ab: ab + 2),
            operation(name='sub2', needs=['c'], provides=['d'], color='red')(lambda c: c - 2)
        ),
        ElseIf(name='elseif', needs=['ab'], provides=['d'], condition_needs=['ab'], condition=lambda ab: ab > 2)(
            operation(name='mul2', needs=['ab'], provides=['d'], color='blue')(lambda ab: ab*10)
        ),
        Else(name='else_less_than_2', needs=['ab'], provides=['d'])(
            operation(name='sub', needs=['ab'], provides=['c'], color='red')(lambda ab: ab - 1),
            operation(name='add2', needs=['c'], provides=['d'], color='red')(lambda c: c + 1)
        ),
        operation(name='div', needs=['d'], provides=['e'], color='blue')(lambda d: d/2)
    )

    res = graph({'a': 1, 'b': 3, 'i': 3}, color='red')
    assert res == {'ab': 3}

    res.update({'i': 3})
    res2 = graph(res, color='blue')
    assert res2 == {'d': 30, 'e': 15.0}


def test_type_checking():

    def abspow(a, p=3):
        c = abs(a) ** p
        return c

    try:
        graph = compose(name="graph")(
           operation(name="mul1", needs=[Var("a", int), Var("b", int)], provides=[Var("ab", int)])(mul),
           operation(name="sub1", needs=[Var("a", float), Var("ab", float)], provides=[Var("a_minus_ab", float)])(sub),
           operation(name="abspow1", needs=[Var("a_minus_ab", float)], provides=[Var("abs_a_minus_ab_cubed", float)], params={"p": 3})(abspow)
        )
    except TypeError as e:
        pass

    graph = compose(name="graph")(
        operation(name="mul1", needs=[Var("a", int), Var("b", int)], provides=[Var("ab", int)])(mul),
        operation(name="sub1", needs=[Var("a", int), Var("ab", int)], provides=[Var("a_minus_ab", int)])(sub),
        operation(name="abspow1", needs=[Var("a_minus_ab", int), Var("p", int, optional=True)], provides=[Var("abs_a_minus_ab_cubed", int)])(abspow)
    )

    out = graph({'a': 2, 'b': 5})
    assert out == {'abs_a_minus_ab_cubed': 512, 'a_minus_ab': -8, 'ab': 10}
