# Copyright © 2022 Gurobi Optimization, LLC
"""Building Sub-models with gurobipy"""


class SubModel:
    """Base class for building and representing a sub-model embeded in a gurobipy.Model.

    When instantiating this class, a (sub-)model is created in the provided
    gurobipy.Model.  The instance represents the sub-model that was created,
    and can be removed from it again.  Furthermore, additional information
    about the submodel can be provided, depending on the sub-model creation.

    There are two ways of using this class:

    1. To build a derived class to build a specific model.

    For this, you must create a sub-class that defines the _build_submodel()
    method, where you implement the code that builds the sub-model from
    the provided data in the gurobipy.Model.  The class should be implemented
    in the following way:

    class MySubModel(SubModel):
        def __init__(self, model, ...., **kwargs):
            super().__init(self, model, ...., **kwargs)

        def _build_submodel(self, model, ...., **kwargs):
            ...

    Here "...." can be your desired list of parameters, including possibly
    *args.

    To create your sub-model in a gurobipy.Model object, simply instantiate
    your class:

    my_sub_model = MySubModel(model, ....)

    2. To build a sub-model using a modeling function.

    If you have a function that populates a gurobipy.Model with a sub-model
    from data, you can use this class directly to represent the created
    submodel in the SubModel instance.  The signature of such a modeling
    function must be:

    def my_modeling_function(model, ...., **kwargs):
        ...

    Again, "...." can be your desired list of parameters, including possibly
    *args.

    To use your model building function, e.g. my_modeling_function, with this
    class, simply pass it to the SubModel constructor:

    my_sub_model = SubModel(model, ....., model_function=my_modeling_function, ....)

    Indipendently of how you created your SubModel instance, you can remove
    the sub-model from the gurobipy.Model by calling the remove method:

    my_sub_model.remove()

    You can always pass the named parameter name='my_name' to the consturctors
    of a submodel.  This name will be used to prefix the names of all the
    modeling object that are being created during submodel construction. If
    no name is provided an automatic name is generated from a default name.
    The default name can be specified with the default_name parameter (mostly
    useful with sub-classing); if not it is derived from the class name of
    your derived class, or the name of your model construction function.
    The only exception is the empty string name='' which is not used as a
    prefix.  This is usefull if the full model is built as a SubModel.

    Note, that you can instantiate as many SubModels as needed in one
    gurobipy.Model.  In particular, you can instantiate SubModels from
    within the model construction method or function of a SubModel itself.
    In this case, the automatic name handling will '.'-separated generate
    names following your call hierarchy.

    Parameters
    ----------

    """

    def __init__(
        self,
        grbmodel,
        *args,
        model_function=None,
        default_name=None,
        name=None,
        **kwargs,
    ):
        self._model = None
        self._objects = {}
        self._firstvar = None
        self._lastvar = None
        self._firstconstr = None
        self._lastconstr = None
        self._qconstrs = []
        self._genconstrs = []
        self._sos = []
        self._first_callback = None
        self._last_callback = None
        self._name = None
        self._model_function = model_function

        if default_name is not None:
            self._default_name = default_name
        elif model_function is not None:
            self._default_name = model_function.__name__
        else:
            self._default_name = type(self).__name__

        before = self._open(grbmodel)
        self._objects = self._build_submodel(grbmodel, *args, **kwargs)
        if self._objects is None:
            self._objects = {}
        self._close(before, name)

    def _build_submodel(self, grbmodel, *args, **kwargs):
        """Method to be overridden for generating the model in a sub-class.

        When using SubModel to wrap a modeling function in a SubModel, the default
        implementation of this method simply calls the modeling function.
        """
        return self._model_function(grbmodel, *args, **kwargs)

    class _ModelingData:
        """Class for recording modeling data in a gurobipy.Model object"""

        def __init__(self):
            self.name_handler = None

        def pop_name_handler(self):
            """Remove name handler and return it."""
            name_handler = self.name_handler
            self.name_handler = None
            return name_handler

        def push_name_handler(self, name_handler):
            """install name handler"""
            self.name_handler = name_handler

    class _modelstats:
        """Helper class for recording gurobi model dimensions

        Parameters
        ----------
        model: gp.Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>
            A gurobipy model

        Attributes
        ----------
        numvars: int
            Number of variables in `model`.
        numconstrs: int
            Number of constraints in `model`.
        numsos: int
            Number of SOS constraints in `model`
        numqconstrs: int
            Number of quadratic constraints in `model`
        numgenconstrs: int
            Number of general constraints in `model`
        """

        def __init__(self, model):
            model.update()
            self.numvars = model.numVars
            self.numconstrs = model.numConstrs
            self.numsos = model.numSOS
            self.numqconstrs = model.numQConstrs
            self.numgenconstrs = model.numGenConstrs

    def _record(self, model, before):
        """Record added modeling objects compared to status before."""
        model.update()
        # range of variables
        if model.numvars > before.numvars:
            self._firstvar = model.getVars()[before.numvars]
            self._lastvar = model.getVars()[model.numvars - 1]
        else:
            self._firstvar = None
            self._lastvar = None
        # range of constraints
        if model.numconstrs > before.numconstrs:
            self._firstconstr = model.getConstrs()[before.numconstrs]
            self._lastconstr = model.getConstrs()[model.numconstrs - 1]
        else:
            self._firstconstr = None
            self._lastconstr = None
        # range of Q constraints
        if model.numqconstrs > before.numqconstrs:
            self._qconstrs = model.getQConstrs()[before.numqconstrs : model.numqconstrs]
        else:
            self._qconstrs = []
        # range of GenConstrs
        if model.numgenconstrs > before.numgenconstrs:
            self._genconstrs = model.getGenConstrs()[before.numgenconstrs : model.numgenconstrs]
        else:
            self._genconstrs = []
        # range of SOS
        if model.numsos > before.numsos:
            self._sos = model.getSOSs()[before.numsos : model.numsos]
        else:
            self._sos = []

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

    def _open(self, model):
        """Start registering modeling object that are added to the gurobipy.Model"""
        self._model = model
        try:
            modeling_data = model._modeling_data
        except AttributeError:
            modeling_data = SubModel._ModelingData()
        model._modeling_data = modeling_data

        return (self._modelstats(model), model._modeling_data.pop_name_handler())

    def _close(self, before, name):
        """Finalize addition of modeling objects to the gurobipy.Model object.

        Record all added modeling objects in SubModel and prefix their names.
        """

        class NameHandler:
            """Handle automatic name generation in gp.Model"""

            def __init__(self):
                self.name = {}

            def get_name(self, sub: SubModel):
                """Return a default name for specified submodel sub."""
                name = sub.default_name
                try:
                    num = self.name[name]
                except KeyError:
                    num = 1
                self.name[name] = num + 1
                return f"{name}{num}"

        def prefix_names(model, objs, attr, name):
            """Prefix all modeling object names with name"""
            if len(objs) == 0:
                return
            object_names = model.getAttr(attr, objs)
            new_names = [f"{name}.{obj_name}" for obj_name in object_names]
            model.setAttr(attr, objs, new_names)

        # re-install name handler
        self._model._modeling_data.push_name_handler(before[1])

        # record all newly added modeling objects
        self._record(self._model, before[0])

        # prefix names of newly created modeling objects
        if name != "":
            if name is None:
                name_handler = self._model._modeling_data.name_handler
                if name_handler is None:
                    name_handler = NameHandler()
                    self._model._modeling_data.push_name_handler(name_handler)
                name = name_handler.get_name(self)
            self._name = name
            prefix_names(self._model, self.vars, "VarName", name)
            prefix_names(self._model, self.constrs, "ConstrName", name)
            prefix_names(self._model, self.qconstrs, "QCName", name)
            prefix_names(self._model, self.genconstrs, "GenConstrName", name)
            # SOS can't have a name! :-O
            # prefix_names(self._model, self.sos, "SOSName", name)

    def print_stats(self, file=None):
        """Print statistics about submodel created"""
        name = self._name
        if name == "":
            name = self.default_name

        print(f"Model for {name}:", file=file)
        print(f"{len(self.vars)} variables", file=file)
        print(f"{len(self.constrs)} constraints", file=file)
        qconstr = len(self.qconstrs)
        if qconstr > 0:
            print(f"{qconstr} quadratic constraints", file=file)
        genconstr = len(self.genconstrs)
        if genconstr > 0:
            print(f"{genconstr} general constraints", file=file)
        sosconstr = len(self.sos)
        if sosconstr > 0:
            print(f"{sosconstr} SOS constraints", file=file)

    @property
    def model(self):
        """Access model the submodel is a part of"""
        return self._model

    @property
    def results(self):
        """Access the results from the _build_submodel() method."""
        return self._objects

    @property
    def default_name(self):
        """Access the default name base used for automatic name generation."""
        return self._default_name

    def remove(self):
        """Remove the submodel from the model"""
        if self._model:
            self._model.remove(self.vars)
            self._model.remove(self.constrs)
            self._model.remove(self.qconstrs)
            self._model.remove(self.genconstrs)
            self._model.remove(self.sos)
            if self._first_callback:
                beg = self._model.getCallbackIndex(self._first_callback)
                end = self._model.getCallbackIndex(self._last_callback)
                self._model.removeCallbacks(beg, end)

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
            self._objects = None

    # Methods to give direct access to result dict

    def keys(self):
        """Returns all keys of the results dictionary"""
        return self._objects.keys()

    def __getitem__(self, key):
        return self._objects[key]

    def __iter__(self):
        return self._objects.__iter__()

    def __len__(self):
        return self._objects.__len__()
