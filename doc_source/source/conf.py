# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

from sphinx_pyproject import SphinxConfig

sys.path.insert(0, os.path.abspath("../../src/"))
# -- Project information -----------------------------------------------------
config = SphinxConfig("../../pyproject.toml", globalns=globals())

copyright = "2022, Gurobi Optimization, LLC. All Rights Reserved."
html_logo = "_static/image8.png"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
]

dep_versions = {
    x.split("==")[0]: x.split("==")[1] for x in (Path().resolve().parent.parent / "requirements.tox.txt").read_text().split()
}
rst_epilog = f"""
.. |PandasVersion| replace:: {dep_versions["pandas"]}
.. |TorchVersion| replace:: {dep_versions["torch"]}
.. |SklearnVersion| replace:: {dep_versions["scikit-learn"]}
.. |TensorflowVersion| replace:: {dep_versions["tensorflow"]}
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
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"


myst_enable_extensions = [
    "dollarmath",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
autodoc_member_order = "groupwise"
autodoc_mock_imports = ["torch", "tensorflow"]
nbsphinx_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "Rmd"}],
}
nbsphinx_allow_errors = True
