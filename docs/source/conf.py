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
html_theme = "gurobi_sphinxtheme"
html_title = project + " Manual"
html_theme_options = {
    "version_warning": False,
    "feedback_banner": True,
    "construction_warning": False,
    "sidebar_hide_name": False,
    "footer_icons": [
        {
            "name": "Gurobi",
            "url": "https://www.gurobi.com",
            "html": """
                <svg id="Layer_2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128.5 127.5"><defs><style>.cls-1{fill:#ed3424;}.cls-2{fill:#c61814;}.cls-3{fill:#22222c;}</style></defs><g id="Layer_2-2"><polygon class="cls-2" points="94.5 6.86 59.08 0 12.07 30.33 74.92 49.88 94.5 6.86"/><polygon class="cls-1" points="9.3 34.11 6.36 53.16 0 94.45 77.03 121.14 95.78 127.64 74.33 54.35 9.3 34.11"/><polygon class="cls-2" points="97.79 10.33 78.49 52.75 100.14 126.74 128.5 98.36 97.79 10.33"/></g></svg>
            """,
            "class": "",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/Gurobi/gurobi-machinelearning",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}
html_show_sourcelink = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
autodoc_member_order = "groupwise"
autodoc_mock_imports = ["torch", "tensorflow", "xgboost"]
html_css_files = [
    "gurobi_ml.css",
]
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
