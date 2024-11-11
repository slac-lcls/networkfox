# Copyright 2019 Stanford University
# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.


class Operation(object):
    """
    This is an abstract class representing a data transformation. To use this,
    please inherit from this class and customize the ``.compute`` method to your
    specific application.
    """

    def __init__(self, **kwargs):
        """
        Create a new layer instance.
        Names may be given to this layer and its inputs and outputs. This is
        important when connecting layers and data in a Network object, as the
        names are used to construct the graph.

        :param str name: The name the operation (e.g. conv1, conv2, etc..)

        :param list needs: Names of input data objects this layer requires.

        :param list provides: Names of output data objects this provides.

        :param dict params: A dict of key/value pairs representing parameters
                            associated with your operation. These values will be
                            accessible using the ``.params`` attribute of your object.
                            NOTE: It's important that any values stored in this
                            argument must be pickelable.
        """

        # (Optional) names for this layer, and the data it needs and provides
        self.name = kwargs.get('name')
        self.needs = kwargs.get('needs')
        self.provides = kwargs.get('provides')
        self.params = kwargs.get('params', {})
        self.color = kwargs.get('color', None)
        self.order = 0
        self.metadata = kwargs.get('metadata', {})
        self.warning = None
        # call _after_init as final step of initialization
        self._after_init()

    def __eq__(self, other):
        """
        Operation equality is based on name of layer.
        (__eq__ and __hash__ must be overridden together)
        """
        return bool(self.name is not None and
                    self.name == getattr(other, 'name', None))

    def __lt__(self, other):
        return self.order < other.order

    def __hash__(self):
        """
        Operation equality is based on name of layer.
        (__eq__ and __hash__ must be overridden together)
        """
        return hash(self.name)

    def compute(self, inputs):
        """
        This method must be implemented to perform this layer's feed-forward
        computation on a given set of inputs.
        :param list inputs:
            A list of :class:`Data` objects on which to run the layer's
            feed-forward computation.
        :returns list:
            Should return a list of :class:`Data` objects representing
            the results of running the feed-forward computation on
            ``inputs``.
        """

        raise NotImplementedError

    def _compute(self, named_inputs, outputs=None):
        inputs = [named_inputs[d] for d in self.needs]
        results = self.compute(inputs)

        results = zip(self.provides, results)
        if outputs:
            outputs = set(outputs)
            results = filter(lambda x: x[0] in outputs, results)

        return dict(results)

    def _after_init(self):
        """
        This method is a hook for you to override. It gets called after this
        object has been initialized with its ``needs``, ``provides``, ``name``,
        and ``params`` attributes. People often override this method to implement
        custom loading logic required for objects that do not pickle easily, and
        for initialization of c++ dependencies.
        """
        pass

    def __repr__(self):
        """
        Display more informative names for the Operation class
        """
        return u"%s(name='%s', needs=%s, provides=%s)" % \
            (self.__class__.__name__,
             self.name,
             self.needs,
             self.provides)


class NetworkOperation(Operation):
    def __init__(self, **kwargs):
        self.net = kwargs.pop('net')
        Operation.__init__(self, **kwargs)

    def _compute(self, named_inputs, outputs=None, color=None, pool=None):
        return self.net.compute(outputs, named_inputs, color, pool)

    def __call__(self, *args, **kwargs):
        return self._compute(*args, **kwargs)

    def plot(self, filename=None, show=False):
        return self.net.plot(self.name, filename=filename, show=show)

    def times(self):
        return self.net.times

    def warnings(self):
        return self.net.warnings

    def node_metadata(self):
        return {node.name: node.metadata for node in self.net.graph.nodes() if hasattr(node, "metadata")}


class Control(Operation):

    def __init__(self, **kwargs):
        if not all(isinstance(arg, Var) for arg in kwargs['needs']):
            needs = [Var(arg) for arg in kwargs['needs']]
            kwargs['needs'] = needs

        if not all(isinstance(arg, Var) for arg in kwargs['provides']):
            provides = [Var(arg) for arg in kwargs['provides']]
            kwargs['provides'] = provides

        self.graph = None
        super(Control, self).__init__(**kwargs)

    def __repr__(self):
        """
        Display more informative names for the Operation class
        """
        if hasattr(self, 'condition_needs'):
            return u"%s(name='%s', needs=%s, provides=%s, condition_needs=%s)" % \
                (self.__class__.__name__,
                 self.name,
                 self.needs,
                 self.provides,
                 self.condition_needs)
        else:
            return super(Control, self).__repr__()


class Var(object):
    """
    Class for specifying optional types for inputs and outputs of graph nodes.
    """

    def __init__(self, name, type=object, optional=False):
        self.name = name
        self.type = type
        self.optional = optional

    def __repr__(self):
        return 'Var(name=%s, type=%s, optional=%s)' % (self.name, self.type, self.optional)

    def __eq__(self, other):
        """
        Operation equality is based on name of layer.
        (__eq__ and __hash__ must be overridden together)
        """
        return bool((self.name is not None and
                    self.name == getattr(other, 'name', None)) and
                    self.type is not None and
                    self.type == getattr(other, 'type', None))

    def __hash__(self):
        return hash(self.name)
