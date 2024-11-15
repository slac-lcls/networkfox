# Copyright 2019 Stanford University
# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

from itertools import chain

from .base import Operation, NetworkOperation, Var, Control
from .network import Network
from .modifiers import optional, GraphWarning


class FunctionalOperation(Operation):
    def __init__(self, **kwargs):
        self.fn = kwargs.pop('fn')
        Operation.__init__(self, **kwargs)

    def _compute(self, named_inputs, outputs=None):
        self.warning = None
        inputs = [named_inputs[d.name] for d in self.needs if not d.optional]

        # Find any optional inputs in named_inputs.  Get only the ones that
        # are present there, no extra `None`s.
        optionals = {n.name: named_inputs[n.name] for n in self.needs if n.optional and n.name in named_inputs}
        # Combine params and optionals into one big glob of keyword arguments.
        kwargs = {k: v for d in (self.params, optionals) for k, v in d.items()}

        try:
            result = self.fn(*inputs, **kwargs)
        except GraphWarning as e:
            result = None
            e.node_name = self.name
            e.metadata = self.metadata
            self.warning = e
        except Exception as e:
            e.node_name = self.name
            e.metadata = self.metadata
            raise e

        if len(self.provides) == 1:
            result = [result]

        if result:
            result = zip(map(lambda arg: arg.name, self.provides), result)
            if outputs:
                outputs = set(outputs)
                result = filter(lambda x: x[0] in outputs, result)

            return dict(result)
        else:
            return {}

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class operation(Operation):
    """
    This object represents an operation in a computation graph.  Its
    relationship to other operations in the graph is specified via its
    ``needs`` and ``provides`` arguments.

    :param function fn:
        The function used by this operation.  This does not need to be
        specified when the operation object is instantiated and can instead
        be set via ``__call__`` later.

    :param str name:
        The name of the operation in the computation graph.

    :param list needs:
        Names of input data objects this operation requires.  These should
        correspond to the ``args`` of ``fn``.

    :param list provides:
        Names of output data objects this operation provides.

    :param dict params:
        A dict of key/value pairs representing constant parameters
        associated with your operation.  These can correspond to either
        ``args`` or ``kwargs`` of ``fn`.

    :param str color:
        A color for the node in the computation graph.
    """

    def __init__(self, fn=None, **kwargs):
        self.fn = fn
        Operation.__init__(self, **kwargs)

    def _normalize_kwargs(self, kwargs):

        # Allow single value for needs parameter
        if 'needs' in kwargs and type(kwargs['needs']) == str:
            assert kwargs['needs'], "empty string provided for `needs` parameters"
            kwargs['needs'] = [Var(kwargs['needs'])]

        if not all(isinstance(arg, Var) for arg in kwargs['needs']):
            needs = []

            for arg in kwargs['needs']:
                var = Var(arg)

                if isinstance(arg, optional):
                    var.optional = True

                needs.append(var)

            kwargs['needs'] = needs

        # Allow single value for provides parameter
        if 'provides' in kwargs and type(kwargs['provides']) == str:
            assert kwargs['provides'], "empty string provided for `needs` parameters"
            kwargs['provides'] = [Var(kwargs['provides'])]

        if not all(isinstance(arg, Var) for arg in kwargs['provides']):
            provides = [Var(arg) for arg in kwargs['provides']]
            kwargs['provides'] = provides

        assert kwargs['name'], "operation needs a name"
        assert type(kwargs['needs']) == list, "no `needs` parameter provided"
        assert type(kwargs['provides']) == list, "no `provides` parameter provided"
        assert hasattr(kwargs['fn'], '__call__'), "operation was not provided with a callable"

        if type(kwargs['params']) is not dict:
            kwargs['params'] = {}

        if 'color' in kwargs and type(kwargs['color']) == str:
            assert kwargs['color'], "empty string provided for `color` parameters"

        return kwargs

    def __call__(self, fn=None, **kwargs):
        """
        This enables ``operation`` to act as a decorator or as a functional
        operation, for example::

            @operator(name='myadd1', needs=['a', 'b'], provides=['c'])
            def myadd(a, b):
                return a + b

        or::

            def myadd(a, b):
                return a + b
            operator(name='myadd1', needs=['a', 'b'], provides=['c'])(myadd)

        :param function fn:
            The function to be used by this ``operation``.

        :return:
            Returns an operation class that can be called as a function or
            composed into a computation graph.
        """

        if fn is not None:
            self.fn = fn

        total_kwargs = {}
        total_kwargs.update(vars(self))
        total_kwargs.update(kwargs)
        total_kwargs = self._normalize_kwargs(total_kwargs)

        return FunctionalOperation(**total_kwargs)

    def __repr__(self):
        """
        Display more informative names for the Operation class
        """
        return u"%s(name='%s', needs=%s, provides=%s, fn=%s)" % \
            (self.__class__.__name__,
             self.name,
             self.needs,
             self.provides,
             self.fn.__name__)


class compose(object):
    """
    This is a simple class that's used to compose ``operation`` instances into
    a computation graph.

    :param str name:
        A name for the graph being composed by this object.

    :param bool merge:
        If ``True``, this compose object will attempt to merge together
        ``operation`` instances that represent entire computation graphs.
        Specifically, if one of the ``operation`` instances passed to this
        ``compose`` object is itself a graph operation created by an
        earlier use of ``compose`` the sub-operations in that graph are
        compared against other operations passed to this ``compose``
        instance (as well as the sub-operations of other graphs passed to
        this ``compose`` instance).  If any two operations are the same
        (based on name), then that operation is computed only once, instead
        of multiple times (one for each time the operation appears).
    """

    def __init__(self, name=None, merge=False):
        assert name, "compose needs a name"
        self.name = name
        self.merge = merge

    def __call__(self, *operations):
        """
        Composes a collection of operations into a single computation graph,
        obeying the ``merge`` property, if set in the constructor.

        :param operations:
            Each argument should be an operation instance created using
            ``operation``.

        :return:
            Returns a special type of operation class, which represents an
            entire computation graph as a single operation.
        """
        assert len(operations), "no operations provided to compose"

        # If merge is desired, deduplicate operations before building network
        if self.merge:
            merge_set = set()
            for op in operations:
                if isinstance(op, NetworkOperation):
                    net_ops = filter(lambda x: isinstance(x, Operation), op.net.steps)
                    merge_set.update(net_ops)
                else:
                    merge_set.add(op)
            operations = list(merge_set)

        def order_preserving_uniquifier(seq, seen=None):
            seen = seen if seen else set()
            seen_add = seen.add
            return [x for x in seq if not (x in seen or seen_add(x))]

        provides = order_preserving_uniquifier(chain(*[op.provides for op in operations]))
        needs = order_preserving_uniquifier(chain(*[op.needs for op in operations]), set(provides))

        # compile network
        net = Network()

        control_nodes = list(filter(lambda op: isinstance(op, Control), operations))
        for idx, control_node in enumerate(control_nodes):
            if isinstance(control_node, If):
                try:
                    if isinstance(control_nodes[idx+1], Else):
                        control_node.Else = control_nodes[idx+1]
                except IndexError:
                    continue
            elif isinstance(control_node, Else):
                if not isinstance(control_nodes[idx-1], If):
                    raise Exception("Else not preceded by If")
                control_node.If = control_nodes[idx-1]

        for op in operations:
            net.add_op(op)
        net.compile()

        return NetworkOperation(name=self.name, needs=needs, provides=provides, params={}, net=net)


class If(Control):

    def __init__(self, condition_needs, condition, **kwargs):
        super(If, self).__init__(**kwargs)
        self.order = 1
        self.condition_needs = condition_needs
        self.condition = condition
        self.computed_condition = False
        self.Else = None

    def __call__(self, *args):
        self.graph = compose(name=self.name)(*args)
        return self

    def _compute_condition(self, named_inputs):
        inputs = [named_inputs[d] for d in self.condition_needs]
        self.computed_condition = self.condition(*inputs)
        return self.computed_condition

    def _compute(self, named_inputs, color=None):
        return self.graph(named_inputs, color=color)


class Else(Control):

    def __init__(self, **kwargs):
        super(Else, self).__init__(**kwargs)
        self.order = 2
        self.If = None

    def __call__(self, *args):
        self.graph = compose(name=self.name)(*args)
        return self

    def _compute(self, named_inputs, color=None):
        return self.graph(named_inputs, color=color)
