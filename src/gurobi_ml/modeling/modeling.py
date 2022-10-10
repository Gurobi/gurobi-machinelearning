"""Modelling support for gurobipy

Building models with gurobipy is done by adding variables and constraints
to a gurobipy.Model object.  To assist building a library of reusable and
combinable modeling constructs, this module provides two decorators for
your modeling function,

    @prefix_names(type_name: str)

and

    @make_submodel(type_name: str)

Both rely on any modeling construct to be implemented in a function or
class as follows.  A modeling function must be a function with the following
signature:

    @prefix_names('Type')
    def modeling_function(model: gurobipy.Model, ...):
        pass

That is, a gurobipy.Model must be the first parameter of the function.
The rest of the parameters are freely choosable, except that it must not
include a parameter called 'name'.  This will be added to the function
by the decorator as explained below.

The same decorator can be used to decorate a class as follows

    @prefix_names('Type')
    class ModelingClass:
        def __init__(self, gurobipy.Model, ....):
            pass

In this case the decorator is applied to constructor of the class.  As a
consequence, the restriction to the __init__ method.  That is, after the
self parameter a gurobipy.Model must be the first parameter of the function.
The rest of the parameters are freely choosable, except that it must not
include a parameter called 'name'.  This will be added to the function
by the decorator as explained below.

The puropose of a modeling function or class is to add variables and/or
constraints to the gurobipy.Model object passed to it.  The set of these
additions is referred to as a submodel, even though it could well be the
full model.

When decorating a modeling function or class with the @prefix_names(type_name)
decorator, the function or class constructor receives an additional named
parameter called 'name'.  This parameter is used in the following way to
prefix the names of all modeling objects in the submodel:

 - If name is the empty string ("" or ''), the names remain unaffected as
   if the undecorated function or constructor was called.
 - If name is None, all names of the modeling objects of the submodel are
   prefixed with a default prefix f'{default_name}.' where default_name is
   generated from the type_name passed to the decorator and a consequtive
   number.
 - Otherwise, all names of the modeling objects of the submodel are prefixed
   with the prefix f{name.}.

The @make_submodel decorator is an extension of the @prefix_names in that it
also affects the return value of the decorated function or constructor.  It
will change the type of the returned value to be also derived from SubModel
and install the necessary data in an additional attribute _submodel.  This
allows you to query information about the submodel that was added to the
gurobipy.Model object or remove it again.

Of course, decorators can also be called directly, i.e. to build a decorated
version of an existing modeling function or class to which one has no access
to the sources.  For instane for a given modeling function my_function, this
would be done with the following line:

    my_decorated_function = prefix_names('MyType')(my_function)
"""

import inspect
from typing import Callable

import gurobipy as gp


class SubModel:
    """A way to capture what has been added to a model.

    What get's recorded are all the native modeling object added to a model
    from the moment to the construction of a SubModel to when submodel.close()
    is executed.  Method close() cannot be executed more than once.

    Once a submodel has been close()ed, the variables etc can be queried using
    the properties var, etc.
    """

    class Data:
        """Internal class to record submodel data for a decorated modeling function or class.

        When using the @make_submodel decorator, an instance of this class is added as
        attribute _submodel to the return value of the modeling function or constructor
        and the type of the return value is upgraded to implement the SubModel interface.
        """

        def open(self, model: gp.Model, type_name: str):
            self._model = model
            self._type_name = type_name

            model.update()

            modeling = Modeling(model)._modeling
            modeling.push_name_handler()

            self.numvars = model.numVars
            self.numconstrs = model.numConstrs
            self.numsos = model.numSOS
            self.numqconstrs = model.numQConstrs
            self.numgenconstrs = model.numGenConstrs
            # self.numcallbacks = modeling.numCallbacks

        def close(self, name: str = None):
            modeling = Modeling(self._model)._modeling

            self._model.update()

            if self._model.numvars > self.numvars:
                self._firstvar = self._model.getVars()[self.numvars]
                self._lastvar = self._model.getVars()[self._model.numvars - 1]
            else:
                self._firstvar = None
                self._lastvar = None
            # range of constraints
            if self._model.numconstrs > self.numconstrs:
                self._firstconstr = self._model.getConstrs()[self.numconstrs]
                self._lastconstr = self._model.getConstrs()[self._model.numconstrs - 1]
            else:
                self._firstconstr = None
                self._lastconstr = None
            # range of Q constraints
            if self._model.numqconstrs > self.numqconstrs:
                self._qconstrs = self._model.getQConstrs()[self.numqconstrs : self._model.numqconstrs]
            else:
                self._qconstrs = []
            # range of GenConstrs
            if self._model.numgenconstrs > self.numgenconstrs:
                self._genconstrs = self._model.getGenConstrs()[self.numgenconstrs : self._model.numgenconstrs]
            else:
                self._genconstrs = []
            # range of SOS
            if self._model.numsos > self.numsos:
                self._sos = self._model.getSOSs()[self.numsos : self._model.numsos]
            else:
                self._sos = []
            # range of Callbacks
            # if self._model._modeling.numcallbacks > self.numcallbacks:
            #     self._first_callback = modeling.getCallback(self.numcallbacks, self.numcallbacks)[0]
            #     self._last_callback  = modeling.getCallback(modeling.numcallbacks - 1, modeling.numcallbacks - 1)[0]

            del self.numvars
            del self.numconstrs
            del self.numsos
            del self.numqconstrs
            del self.numgenconstrs

            modeling.pop_name_handler()
            if name != "":
                Modeling(self._model)._modeling.prefix_names(self, name)
            else:
                self._name = name

        def removeFromModel(self):
            """Remove the submodel from the gurobipy.Model object it is part of."""
            if self._model:
                self._model.remove(self.vars)
                self._model.remove(self.constrs)
                self._model.remove(self.qconstrs)
                self._model.remove(self.genconstrs)
                self._model.remove(self.sos)
                if self._first_callback:
                    beg = self._model._modeling.getCallbackIndex(self._first_callback)
                    end = self._model._modeling.getCallbackIndex(self._last_callback)
                    self._model._modeling.removeCallbacks(beg, end)

                self._firstvar = None
                self._lastvar = None
                self._firstconstr = None
                self._lastconstr = None
                self._qconstrs = []
                self._genconstrs = []
                self._sos = []
                self._first_callback = None
                self._last_callback = None
                self._model = None
                self._name = None

        @property
        def type(self):
            """Return the submodel type."""
            return self._type_name

        @property
        def name(self):
            """Return the submodel name."""
            if self._name is None:
                return self.type
            return self._name

        @property
        def model(self):
            """Return the gurobipy.Model the submodel belongs to."""
            return self._model

        @property
        def vars(self):
            """Return the list of variables in the submodel."""
            if self._firstvar:
                return self._model.getVars()[self._firstvar.index : self._lastvar.index + 1]
            return []

        @property
        def constrs(self):
            """Return the list of linear constraints in the submodel."""
            if self._firstconstr:
                return self._model.getConstrs()[self._firstconstr.index : self._lastconstr.index + 1]
            return []

        @property
        def qconstrs(self):
            """Return the list of quadratic constraints in the submodel."""
            return self._qconstrs

        @property
        def genconstrs(self):
            """Return the list of general constraints in the submodel."""
            return self._genconstrs

        @property
        def sos(self):
            """Return the list of SOS constraints in the submodel."""
            return self._sos

        def print_stats(self, file=None):
            """Print statistics about the submodel"""
            name = self.name

            print(f"Model for {name}:", file=file)
            print(f"   {len(self.vars)} variables", file=file)
            print(f"   {len(self.constrs)} constraints", file=file)
            qconstr = len(self.qconstrs)
            if qconstr > 0:
                print(f"   {qconstr} quadratic constraints", file=file)
            genconstr = len(self.genconstrs)
            if genconstr > 0:
                print(f"   {genconstr} general constraints", file=file)
            sosconstr = len(self.sos)
            if sosconstr > 0:
                print(f"   {sosconstr} SOS constraints", file=file)

    def __init__(self, submodel=None):
        """(Copy) Constructor

        Args:
            submodel (SubModel, optional):
                Create a copy of the submodel object.  With this you get
                only the submodel description from a subclass that may also
                include other stuff.
        """
        if submodel is None:
            self._submodel = SubModel.Data()
        else:
            self._submodel = submodel._submodel

    def open(self, model: gp.Model, type_name: str):
        """Start recording additions to model

        All model additions from then on until the invocation of the
        close method are recorded to be part of the submodel

        Args:
            model (gp.Model): the model for which to record additions
            type_name (str):  The type name of the submodel

        NOTE: This method can only be called once on a submodel object.
        """
        self._submodel.open(model, type_name)

    def close(self, name: str = None):
        """Stop recording additions to model

        All model additions since the invocation of the open method
        are made part of this submodel.  Their names are prefixed
        with name - see class doc for details.

        Args:
            name (str, optional): Prefix name for modeling objects
            in this submodel

        NOTE: This method can only be called once on a submodel object.
        """
        self._submodel.close(name)

    def removeFromModel(self):
        """Remove the submodel from the gurobipy.Model object it is part of."""
        self._submodel.removeFromModel()

    @property
    def type(self):
        """Return the submodel type."""
        return self._submodel.type

    @property
    def name(self):
        """Return the submodel name."""
        return self._submodel.name

    @property
    def model(self):
        """Return the gurobipy.Model the submodel belongs to."""
        return self._submodel.model

    @property
    def vars(self):
        """Return the list of variables in the submodel."""
        return self._submodel.vars

    @property
    def constrs(self):
        """Return the list of linear constraints in the submodel."""
        return self._submodel.constrs

    @property
    def qconstrs(self):
        """Return the list of quadratic constraints in the submodel."""
        return self._submodel.qconstrs

    @property
    def genconstrs(self):
        """Return the list of general constraints in the submodel."""
        return self._submodel.genconstrs

    @property
    def sos(self):
        """Return the list of SOS constraints in the submodel."""
        return self._submodel.sos

    def print_stats(self, file=None):
        self._submodel.print_stats(file)

    @staticmethod
    def _make_submodel(type_name: str, make_subclass: bool):
        """Decorator for modeling functions.

        When decorating a modeling function with this decorator, its signature is augmented
        with a name parameter.  This parameter can be used to prefix all the names of the
        modeling objects the modeling function creates upon invocation.  This parameter
        can be used as follows:
        - Passing None for the name parameter, will use default names derived from the
        provided type_name.
        - Passing an empty string will skip any prefixing of names.
        - Otherwise, the priveded name will be used as prefix.
        """

        def wrap(function: Callable):
            def add_submodel(obj, submodel):
                """Turn obj into a SubModel using the data from the provided submodel."""
                if obj is None:
                    return submodel
                if not isinstance(obj, SubModel):
                    obj_class = type(obj)
                    if obj_class in {set, dict, list, tuple, int, float, bool}:
                        new_cls = type(f"{obj_class}_SubModel", (obj_class, SubModel), {"__init__": obj_class.__init__})
                        obj = new_cls(obj)
                    else:
                        new_cls = type(f"{obj_class}_SubModel", (obj_class, SubModel), {})
                        obj.__class__ = new_cls
                obj._submodel = submodel._submodel
                return obj

            def wrapped_function(model, *args, name=None, **kwargs):
                indent = None
                if Modeling(model)._modeling._verbose != 0:
                    indent = model._modeling._indent
                    model._modeling._indent = "   " + indent
                    print(f"{indent}Creating Submodel of type {type_name}:")
                    if model._modeling._verbose > 1:
                        from pprint import pprint

                        if len(args) > 0:
                            print(f"{indent}  Positional Arguments:", end="")
                            print(f"{indent}  ", end="")
                            if model._modeling._verbose > 0:
                                pprint(args, depth=model._modeling._verbose)
                            else:
                                pprint(args)
                        if len(kwargs) > 0:
                            print(f"{indent}  Keyword Arguments:", end="")
                            print(f"{indent}  ", end="")
                            if model._modeling._verbose > 0:
                                pprint(kwargs, depth=model._modeling._verbose)
                            else:
                                pprint(kwargs)

                submodel = SubModel()
                submodel.open(model, type_name)
                result = function(model, *args, **kwargs)
                submodel.close(name)

                if indent is not None:
                    print(f"{indent}  Name:                  {submodel.name}")
                    l = len(submodel.vars)
                    if l > 0:
                        print(f"{indent}  Variables:             {l}")
                    l = len(submodel.constrs)
                    if l > 0:
                        print(f"{indent}  Linear constraints:    {l}")
                    l = len(submodel.qconstrs)
                    if l > 0:
                        print(f"{indent}  Quadratic constraints: {l}")
                    l = len(submodel.genconstrs)
                    if l > 0:
                        print(f"{indent}  General Constraints:   {l}")
                    l = len(submodel.sos)
                    if l > 0:
                        print(f"{indent}  SOS Constraints:       {l}")
                    model._modeling._indent = indent

                if make_subclass:
                    return add_submodel(result, submodel)
                return result

            sig = inspect.signature(function)
            par = list(sig.parameters.values())

            # add 'name' parameter with default argument
            if par[len(par) - 1].kind == inspect.Parameter.VAR_KEYWORD:
                par.insert(len(par) - 1, inspect.Parameter("name", inspect.Parameter.KEYWORD_ONLY, default=None))
            else:
                par.insert(len(par) - 0, inspect.Parameter("name", inspect.Parameter.KEYWORD_ONLY, default=None))

            # install signature, name and documentaion
            wrapped_function.__signature__ = sig.replace(parameters=par)
            if inspect.isclass(function):
                wrapped_function.__doc__ = function.__init__.__doc__
            else:
                wrapped_function.__doc__ = function.__doc__
            wrapped_function.__name__ = function.__name__
            wrapped_function._type_name = type_name

            return wrapped_function

        return wrap


class Modeling:
    """Additional modeling functionality/data to be attached to a gurobipy.Model object.

    When using prefix_names with modeling functions, an instance of this class
    is added as attribute _modeling to the gurobipy.Model object, to perform
    prefixing and bookkeeping of names.
    """

    class Data:
        """Internal data stored in gubobipy.Model._modeling."""

        class NameHandler:
            def __init__(self, previous=None):
                self._names = dict()
                self._previous = previous

            def get_name(self, default_name):
                """Return a default name for specified submodel sub."""
                name = default_name
                try:
                    num = self._names[name]
                except KeyError:
                    num = 1
                self._names[name] = num + 1
                return f"{name}{num}"

        def __init__(self, model: gp.Model):
            self._name_handler = Modeling.Data.NameHandler()
            self._indent = ""
            model._modeling = self

        def pop_name_handler(self):
            """Remove name handler and return it."""
            self._name_handler = self._name_handler._previous

        def push_name_handler(self):
            """install a new name handler"""
            self._name_handler = Modeling.Data.NameHandler(self._name_handler)

        def prefix_names(self, submodel, name: str = None):
            """Prefix the names of the modeling objects of this submodel.

            If an empty string is given as name, no prefixing will happen.
            If the default None is passed as name, a default name derived
            from the submodel type will be used.  Otherwise the specified
            name is use as prefix.
            """

            def prefix_set(model: gp.Model, objs, attr: str, name: str):
                """Prefix all modeling object names with name"""
                if len(objs) == 0:
                    return
                object_names = model.getAttr(attr, objs)
                new_names = [f"{name}.{obj_name}" for obj_name in object_names]
                model.setAttr(attr, objs, new_names)

            if name == "":
                return

            if name is None:
                name = self._name_handler.get_name(submodel._type_name)

            submodel._name = name
            prefix_set(submodel._model, submodel.vars, "VarName", name)
            prefix_set(submodel._model, submodel.constrs, "ConstrName", name)
            prefix_set(submodel._model, submodel.qconstrs, "QCName", name)
            prefix_set(submodel._model, submodel.genconstrs, "GenConstrName", name)
            # SOSs don't have names.
            # prefix_set(submodel.sos, "SOSName", name)

    def __init__(self, model: gp.Model):
        """Get the _modeling object from a gurobipy.Model.

        If it is not already an attribute of the gurobipy.Model object, it will
        be created and added.
        """
        try:
            self._modeling = model._modeling
        except AttributeError:
            self._modeling = model._modeling = Modeling.Data(model)
            self._modeling._indent = ""
            self._modeling._verbose = 0

    def setVerbose(self, level: int):
        """Control verbosity when apply (decorated) modeling functions or constructors."""
        self._modeling._verbose = level


def make_name_prefix(modeling_function: Callable, default_name: str) -> Callable:
    """Create a function that adds name prefixing to modeling_function.

    Args:
        modeling_function (Callable): A modeling function or class
        default_name (str): A string used for generating default names for the
                            modeling objects created by the modeling_function.

    Returns:
         Callable: A function that adds name prefixing to modeling_function
    """
    return SubModel._make_submodel(default_name, False)(modeling_function)


def make_submodel(modeling_function: Callable, type_name: str) -> Callable:
    """Create a function that wraps a modeling_function into a SubModel.

    Args:
        modeling_function (Callable): A modeling function or class
        type_name (str): A name representing the type of model being created

    Returns:
         Callable: A function that creates a submodel for the specified modeling function
    """
    return SubModel._make_submodel(type_name, True)(modeling_function)


def name_prefix(default_name: str):
    """Decorator that promotes a modeling functions with name prefixing.

    Decorating modeling_function with @prefix_names(default_name) is equivalent
    to renaming the modeling_function with:

        modeling_function = make_prefix_names(modeling_function, default_name)
    """
    return SubModel._make_submodel(default_name, False)


def submodel(type_name: str):
    """Decorator that promotes a modeling function return a SubModel.

    Decorating modeling_function with @submodel(type_name) is equivalent to
    renaming the modeling_function with:

        modeling_function = make_submodel(modeling_function, type_name)
    """
    return SubModel._make_submodel(type_name, True)
