"""Building Sub-models with gurobipy"""

# pylint: disable=C0103

def addtomodel(function):
    """Decorator function for SobModels"""

    def wrapper(self, grbmodel, *args, name=None, **kwargs):
        """Register model additions and automatic namehandling
        when building a submodel."""
        self.build(grbmodel, function, *args, obj=self, name=name, **kwargs)

    return wrapper


class SubModel:
    """Representation of a sub-model created by a user sub-model addition"""

    class ModelingData:
        """Class for recording modeling data in a gurAobi model"""

        def __init__(self):
            self.nameHandler = None

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
                self.numCallbacks = model.numCallbacks()
            except:  # pylint: disable=bare-except
                self.numCallbacks = 0

    def __init__(self):
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
            self._QConstrs = model.getQConstrs(
            )[before.numQConstrs: model.numQConstrs]
        else:
            self._QConstrs = []
        # range of GenConstrs
        if model.numGenConstrs > before.numGenConstrs:
            self._GenConstrs = model.getGenConstrs()[
                before.numGenConstrs: model.numGenConstrs
            ]
        else:
            self._GenConstrs = []
        # range of SOS
        if model.numSOS > before.numSOS:
            self._SOSs = model.getSOSs()[before.numSOS: model.numSOS]
        else:
            self._SOSs = []
        # range of Callbacks
        self._firstCallback = None
        self._lastCallback = None
        try:
            if model.numCallbacks() > before.numCallbacks:
                self._firstCallback = self._model.getCallback(
                    before.numCallbacks, before.numCallbacks
                )[0]
                self._lastCallback = self._model.getCallback(
                    model.numCallbacks() - 1, model.numCallbacks() - 1
                )[0]
        except AttributeError:
            pass

    def getVars(self):
        """Return the list of variables in the submodel."""
        if self._firstVar:
            return self._model.getVars()[self._firstVar.index: self._lastVar.index + 1]
        return []

    def getConstrs(self):
        """Return the list of linear constraints in the submodel."""
        if self._firstConstr:
            return self._model.getConstrs()[
                self._firstConstr.index: self._lastConstr.index + 1
            ]
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

    def getCallbacks(self):
        """Return the list of callbacks in the submodel."""
        if self._firstCallback:
            return self._model.getCallbacks(self._firstCallback, self._lastCallback)
        return []

    def _open(self, model):
        # Register modeling data in gurobi model
        self._model = model
        try:
            md = model._modeling_data
        except AttributeError:
            md = SubModel.ModelingData()
            model._modeling_data = md

        # Move name handler out of the way to get
        # new numbering in the new hierarchy level
        nameHandler = model._modeling_data.nameHandler
        model._modeling_data.nameHandler = None

        return (self._modelstats(model), nameHandler)

    def _close(self, before, obj, name):
        class NameHandler:
            """Handle automatic name generation in gp.Model"""

            def __init__(self, model):
                self.model = model
                self.name = dict()

            def getName(self, obj):
                """Return a default name for specified obj"""
                if callable(obj):
                    name = obj.__name__
                else:
                    name = type(obj).__name__
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
        self._model._modeling_data.nameHandler = before[1]

        # record all newly added modeling objects
        self._record(self._model, before[0])

        # prefix names of newly created modeling objects
        if name is None:
            nameHandler = self._model._modeling_data.nameHandler
            if nameHandler is None:
                nameHandler = NameHandler(self._model)
                self._model._modeling_data.nameHandler = nameHandler
            name = nameHandler.getName(obj)
        if name != "":
            prefix_names(self.getVars(), "VarName", name)
            prefix_names(self.getConstrs(), "ConstrName", name)
            prefix_names(self.getQConstrs(), "QCName", name)
            prefix_names(self.getGenConstrs(), "GenConstrName", name)
            prefix_names(self.getSOSs(), "SOSName", name)

    def build(self, grbmodel, function, *args, obj=None, name=None, **kwargs):
        """Register model additions and automatic namehandling
        when building a submodel."""
        assert callable(function)
        before = self._open(grbmodel)
        if obj is not None:
            self._objects = function(obj, grbmodel, *args, **kwargs)
        else:
            self._objects = function(grbmodel, *args, **kwargs)
        self._close(before, self, name)

    def getModel(self):
        """Access model the submodel is a part of"""
        return self._model

    def getObjects(self):
        """Access the objects dictionary returned by sub-model constructor"""
        return self._objects

    def __getitem__(self, key):
        return self._objects[key]

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
