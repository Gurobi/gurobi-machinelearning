# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

from sphinx_gallery.sorting import FileNameSortKey
from sphinx_pyproject import SphinxConfig

sys.path.insert(0, os.path.abspath("../../src/"))
# -- Project information -----------------------------------------------------
config = SphinxConfig("../../pyproject.toml", globalns=globals())

project = "Gurobi Machine Learning"
copyright = (
    f"{datetime.datetime.now().year}, Gurobi Optimization, LLC. All Rights Reserved."
)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.imgconverter",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autosectionlabel",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
    "sphinx_design",
]


def get_versions(file: Path):
    new_dict = {}
    for line in file.read_text().splitlines():
        try:
            package, version = line.split("==")
            new_dict[package] = version
        except ValueError:
            pass  # Skip lines that don't split into exactly two items

    return new_dict


root_path = Path().resolve().parent.parent
dep_versions = {
    k: v
    for k, v in get_versions(root_path / "requirements.txt").items()
    if k == "gurobipy"
}  # get only gurobipy from requirements.txt
dep_versions |= get_versions(root_path / "requirements.tox.txt")
dep_versions |= get_versions(root_path / "requirements.keras.txt")
dep_versions |= get_versions(root_path / "requirements.pytorch.txt")
dep_versions |= get_versions(root_path / "requirements.sklearn.txt")
dep_versions |= get_versions(root_path / "requirements.pandas.txt")
dep_versions |= get_versions(root_path / "requirements.xgboost.txt")
dep_versions |= get_versions(root_path / "requirements.lightgbm.txt")


VARS_SHAPE = """See :py:func:`add_predictor_constr <gurobi_ml.add_predictor_constr>` for acceptable values for input_vars and output_vars"""
CLASS_SHORT = (
    """Stores the changes to :gurobipy:`model` for formulating the predictor."""
)


rst_epilog = f"""
.. |GurobiVersion| replace:: {dep_versions["gurobipy"]}
.. |NumpyVersion| replace:: {dep_versions["numpy"]}
.. |ScipyVersion| replace:: {dep_versions["scipy"]}
.. |PandasVersion| replace:: {dep_versions["pandas"]}
.. |TorchVersion| replace:: {dep_versions["torch"]}
.. |SklearnVersion| replace:: {dep_versions["scikit-learn"]}
.. |TensorflowVersion| replace:: {dep_versions["tensorflow"]}
.. |XGBoostVersion| replace:: {dep_versions["xgboost"]}
.. |LightGBMVersion| replace:: {dep_versions["lightgbm"]}
.. |VariablesDimensionsWarn| replace:: {VARS_SHAPE}
.. |ClassShort| replace:: {CLASS_SHORT}
"""

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "xgb": ("https://xgboost.readthedocs.io/en/stable/", None),
    "lightgbm": ("https://lightgbm.readthedocs.io/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

autodoc_default_options = {
    "show-inheritance": True,
}

autoclass_content = "class"

numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    "DataFrame": "pandas.DataFrame",
    "Series": "pandas.Series",
    "Index": "pandas.Index",
}
numpydoc_show_inherited_class_members = {}
numpydoc_xref_ignore = {"optional", "or", "of"}
numpydoc_class_members_toctree = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "text": "GUROBI Machine Learning",
        #     "image_light": "_static/gurobi.png",
        #     "image_dark": "_static/gurobi.png",
        #     "alt_text": "Gurobi home",
    },
    "navbar_align": "content",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Gurobi/Gurobi-machinelearning",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Gurobi",
            "url": "https://www.gurobi.com",
            "icon": "_static/gurobi.png",
            "type": "local",
        },
    ],
}
html_show_sourcelink = False

myst_enable_extensions = [
    "dollarmath",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
autodoc_member_order = "groupwise"
autodoc_mock_imports = ["torch", "tensorflow", "xgboost"]

bibtex_bibfiles = ["refs.bib"]

extlinks_detect_hardcoded_links = True
extlinks = {
    "issue": ("https://github.com/Gurobi/gurobi-machinelearning/issues/%s", "issue %s"),
    "gurobipy": (
        "https://www.gurobi.com/documentation/current/refman/py_%s.html",
        "gurobipy %s",
    ),
    "pypi": ("https://pypi.org/project/%s/", "%s"),
}

sphinx_gallery_conf = {
    "examples_dirs": ["../examples", "../examples_userguide"],
    "gallery_dirs": ["auto_examples", "auto_userguide"],
    "filename_pattern": "/example",
    "within_subsection_order": FileNameSortKey,
    "reference_url": {
        # The module you locally document uses None
        "gurobi_ml": None,
    },
}

# -- Options for LaTeX output -----------------------------------------------------
latex_logo = "_static/gurobi.png"

latex_elements = {
    "preamble": r"""
    \newcommand\sphinxbackoftitlepage{%
Copyright(c), 2023, Gurobi Optimization, LLC. All Rights Reserved.
}
    """,
}
