"""Building Sub-models with gurobipy"""

# pylint: disable=C0103


class SubModel:
    """Base class for building and representing a sub-model of a gurobipy.Model.

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
        self._firstVar = None
        self._lastVar = None
        self._firstConstr = None
        self._lastConstr = None
        self._QConstrs = []
        self._GenConstrs = []
        self._SOSs = []
        self._firstCallback = None
        self._lastCallback = None
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
        """Helper class for recording gurobi model dimensions"""

        def __init__(self, model):
            model.update()
            self.numVars = model.numVars
            self.numConstrs = model.numConstrs
            self.numSOS = model.numSOS
            self.numQConstrs = model.numQConstrs
            self.numGenConstrs = model.numGenConstrs
            try:
                self.numCallbacks = model.numCallbacks
            except AttributeError:
                self.numCallbacks = 0

    def _record(self, model, before):
        """Record added modeling objects compared to status before."""
        model.update()
        # range of variables
        if model.numVars > before.numVars:
            self._firstVar = model.getVars()[before.numVars]
            self._lastVar = model.getVars()[model.numVars - 1]
        else:
            self._firstVar = None
            self._lastVar = None
        # range of constraints
        if model.numConstrs > before.numConstrs:
            self._firstConstr = model.getConstrs()[before.numConstrs]
            self._lastConstr = model.getConstrs()[model.numConstrs - 1]
        else:
            self._firstConstr = None
            self._lastConstr = None
        # range of Q constraints
        if model.numQConstrs > before.numQConstrs:
            self._QConstrs = model.getQConstrs()[before.numQConstrs : model.numQConstrs]
        else:
            self._QConstrs = []
        # range of GenConstrs
        if model.numGenConstrs > before.numGenConstrs:
            self._GenConstrs = model.getGenConstrs()[before.numGenConstrs : model.numGenConstrs]
        else:
            self._GenConstrs = []
        # range of SOS
        if model.numSOS > before.numSOS:
            self._SOSs = model.getSOSs()[before.numSOS : model.numSOS]
        else:
            self._SOSs = []
        # range of Callbacks
        self._firstCallback = None
        self._lastCallback = None
        try:
            if model.numCallbacks > before.numCallbacks:
                self._firstCallback = self._model.getCallback(before.numCallbacks, before.numCallbacks)[0]
                self._lastCallback = self._model.getCallback(model.numCallbacks - 1, model.numCallbacks - 1)[0]
        except AttributeError:
            pass

    def getVars(self):
        """Return the list of variables in the submodel."""
        if self._firstVar:
            return self._model.getVars()[self._firstVar.index : self._lastVar.index + 1]
        return []

    def getConstrs(self):
        """Return the list of linear constraints in the submodel."""
        if self._firstConstr:
            return self._model.getConstrs()[self._firstConstr.index : self._lastConstr.index + 1]
        return []

    def getQConstrs(self):
        """Return the list of quadratic constraints in the submodel."""
        return self._QConstrs

    def getGenConstrs(self):
        """Return the list of general constraints in the submodel."""
        return self._GenConstrs

    def getSOSs(self):
        """Return the list of SOS constraints in the submodel."""
        return self._SOSs

    def _open(self, model):
        """Start registering modeling object that are added to the gurobipy.Model"""
        self._model = model
        try:
            md = model._modeling_data
        except AttributeError:
            md = SubModel._ModelingData()
        model._modeling_data = md

        return (self._modelstats(model), model._modeling_data.pop_name_handler())

    def _close(self, before, name):
        """Finalize addition of modeling objects to the gurobipy.Model object.

        Record all added modeling objects in SubModel and prefix their names.
        """

        class NameHandler:
            """Handle automatic name generation in gp.Model"""

            def __init__(self):
                self.name = {}

            def getName(self, sub: SubModel):
                """Return a default name for specified submodel sub."""
                name = sub.defaultName
                try:
                    num = self.name[name]
                except KeyError:
                    num = 1
                self.name[name] = num + 1
                return f"{name}{num}"

        def prefix_names(objs, attr, name):
            """Prefix all modeling object names with name"""
            for o in objs:
                nm = o.getAttr(attr)
                o.setAttr(attr, f"{name}.{nm}")

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
                name = name_handler.getName(self)

            prefix_names(self.getVars(), "VarName", name)
            prefix_names(self.getConstrs(), "ConstrName", name)
            prefix_names(self.getQConstrs(), "QCName", name)
            prefix_names(self.getGenConstrs(), "GenConstrName", name)
            prefix_names(self.getSOSs(), "SOSName", name)

    @property
    def model(self):
        """Access model the submodel is a part of"""
        return self._model

    @property
    def results(self):
        """Access the results from the _build_submodel() method."""
        return self._objects

    @property
    def defaultName(self):
        """Access the default name base used for automatic name generation."""
        return self._default_name

    def remove(self):
        """Remove the submodel from the model"""
        if self._model:
            self._model.remove(self.getVars())
            self._model.remove(self.getConstrs())
            self._model.remove(self._QConstrs)
            self._model.remove(self._GenConstrs)
            self._model.remove(self._SOSs)
            if self._firstCallback:
                beg = self._model.getCallbackIndex(self._firstCallback)
                end = self._model.getCallbackIndex(self._lastCallback)
                self._model.removeCallbacks(beg, end)

            self._firstVar = None
            self._lastVar = None
            self._firstConstr = None
            self._lastConstr = None
            self._QConstrs = []
            self._GenConstrs = []
            self._SOSs = []
            self._firstCallback = None
            self._lastCallback = None
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
