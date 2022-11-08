# Copyright © 2022 Gurobi Optimization, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Building Sub-models with gurobipy"""


class SubModel:
    """Base class for building and representing a sub-model embedded in a gurobipy.Model.

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

    Independently of how you created your SubModel instance, you can remove
    the sub-model from the gurobipy.Model by calling the remove method:

    my_sub_model.remove()

    You can always pass the named parameter name='my_name' to the constructors
    of a submodel.  This name will be used to prefix the names of all the
    modeling objects that are being created during submodel construction. If
    no name is provided an automatic name is generated from a default name.
    The default may be already specified as an attribute (mostly
    useful with sub-classing); if not, it is derived from the class name of
    your derived class, or the name of your model construction function.
    The only exception is the empty string name='' which is not used as a
    prefix.  This is useful if the full model is built as a SubModel.

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
        gp_model,
        *args,
        model_function=None,
        name=None,
        **kwargs,
    ):
        self._gp_model = None
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

        if not hasattr(self, "_default_name"):
            self._default_name = type(self).__name__

        before = self._open(gp_model)
        self._objects = self._build_submodel(gp_model, *args, **kwargs)
        if self._objects is None:
            self._objects = {}
        self._close(before, name)

    def _build_submodel(self, gp_model, *args, **kwargs):
        """Method to be overridden for generating the model in a sub-class.

        When using SubModel to wrap a modeling function in a SubModel, the default
        implementation of this method simply calls the modeling function.
        """
        return self._model_function(gp_model, *args, **kwargs)

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
        gp_model: gp.Model <https://www.gurobi.com/documentation/9.5/refman/py_model.html>
            A gurobipy model

        Attributes
        ----------
        numvars: int
            Number of variables in `gp_model`.
        numconstrs: int
            Number of constraints in `gp_model`.
        numsos: int
            Number of SOS constraints in `gp_model`
        numqconstrs: int
            Number of quadratic constraints in `gp_model`
        numgenconstrs: int
            Number of general constraints in `gp_model`
        """

        def __init__(self, gp_model):
            gp_model.update()
            self.numvars = gp_model.numVars
            self.numconstrs = gp_model.numConstrs
            self.numsos = gp_model.numSOS
            self.numqconstrs = gp_model.numQConstrs
            self.numgenconstrs = gp_model.numGenConstrs

    def _record(self, gp_model, before):
        """Record added modeling objects compared to status before."""
        gp_model.update()
        # range of variables
        if gp_model.numvars > before.numvars:
            self._firstvar = gp_model.getVars()[before.numvars]
            self._lastvar = gp_model.getVars()[gp_model.numvars - 1]
        else:
            self._firstvar = None
            self._lastvar = None
        # range of constraints
        if gp_model.numconstrs > before.numconstrs:
            self._firstconstr = gp_model.getConstrs()[before.numconstrs]
            self._lastconstr = gp_model.getConstrs()[gp_model.numconstrs - 1]
        else:
            self._firstconstr = None
            self._lastconstr = None
        # range of Q constraints
        if gp_model.numqconstrs > before.numqconstrs:
            self._qconstrs = gp_model.getQConstrs()[before.numqconstrs : gp_model.numqconstrs]
        else:
            self._qconstrs = []
        # range of GenConstrs
        if gp_model.numgenconstrs > before.numgenconstrs:
            self._genconstrs = gp_model.getGenConstrs()[
                before.numgenconstrs : gp_model.numgenconstrs
            ]
        else:
            self._genconstrs = []
        # range of SOS
        if gp_model.numsos > before.numsos:
            self._sos = gp_model.getSOSs()[before.numsos : gp_model.numsos]
        else:
            self._sos = []

    @property
    def vars(self):
        """Return the list of variables in the submodel."""
        if self._firstvar:
            return self._gp_model.getVars()[self._firstvar.index : self._lastvar.index + 1]
        return []

    @property
    def constrs(self):
        """Return the list of linear constraints in the submodel."""
        if self._firstconstr:
            return self._gp_model.getConstrs()[self._firstconstr.index : self._lastconstr.index + 1]
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

    def _open(self, gp_model):
        """Start registering modeling object that are added to the gurobipy.Model"""
        self._gp_model = gp_model
        try:
            modeling_data = gp_model._modeling_data
        except AttributeError:
            modeling_data = SubModel._ModelingData()
        gp_model._modeling_data = modeling_data

        return (self._modelstats(gp_model), gp_model._modeling_data.pop_name_handler())

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

        def prefix_names(gp_model, objs, attr, name):
            """Prefix all modeling object names with name"""
            if len(objs) == 0:
                return
            object_names = gp_model.getAttr(attr, objs)
            new_names = [f"{name}.{obj_name}" for obj_name in object_names]
            gp_model.setAttr(attr, objs, new_names)

        # re-install name handler
        self._gp_model._modeling_data.push_name_handler(before[1])

        # record all newly added modeling objects
        self._record(self._gp_model, before[0])

        # prefix names of newly created modeling objects
        if name != "":
            if name is None:
                name_handler = self._gp_model._modeling_data.name_handler
                if name_handler is None:
                    name_handler = NameHandler()
                    self._gp_model._modeling_data.push_name_handler(name_handler)
                name = name_handler.get_name(self)
            self._name = name
            prefix_names(self._gp_model, self.vars, "VarName", name)
            prefix_names(self._gp_model, self.constrs, "ConstrName", name)
            prefix_names(self._gp_model, self.qconstrs, "QCName", name)
            prefix_names(self._gp_model, self.genconstrs, "GenConstrName", name)
            # SOS can't have a name! :-O
            # prefix_names(self._gp_model, self.sos, "SOSName", name)

    def print_stats(self, file=None):
        """Print statistics about submodel

        This functions prints detailed statistics on the variables
        and constraints that where added to the gp_model using this object.

        Usually derived class reimplement this function to provide more
        details about the structure of the additions (type of ML model,
        layers if it's a neural network,...)

        Parameters
        ---------

        file: None, optional
            Text stream to which output should be redirected. By default sys.stdout.
        """
        name = self._name
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
    def gp_model(self):
        """Access gurobipy model the submodel is a part of"""
        return self._gp_model

    @property
    def default_name(self):
        """Access the default name base used for automatic name generation.

        :meta private:"""
        return self._default_name

    def remove(self):
        """Remove the submodel from the model"""
        if self._gp_model:
            self._gp_model.remove(self.vars)
            self._gp_model.remove(self.constrs)
            self._gp_model.remove(self.qconstrs)
            self._gp_model.remove(self.genconstrs)
            self._gp_model.remove(self.sos)
            if self._first_callback:
                beg = self._gp_model.getCallbackIndex(self._first_callback)
                end = self._gp_model.getCallbackIndex(self._last_callback)
                self._gp_model.removeCallbacks(beg, end)

            self._firstvar = None
            self._lastvar = None
            self._firstconstr = None
            self._lastconstr = None
            self._qconstrs = []
            self._genconstrs = []
            self._sos = []
            self._first_callback = None
            self._last_callback = None
            self._gp_model = None
            self._objects = None
