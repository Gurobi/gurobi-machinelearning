[![build and test](https://github.com/Gurobi/gurobi-machinelearning/actions/workflows/push.yml/badge.svg?branch=main)](https://github.com/Gurobi/gurobi-machinelearning/actions/workflows/push.yml?query=branch%3Amain++)
[![build wheel](https://github.com/Gurobi/gurobi-machinelearning/actions/workflows/build_wheel.yml/badge.svg?branch=main)](https://github.com/Gurobi/gurobi-machinelearning/actions/workflows/build_wheel.yml?query=branch%3Amain++)
![Python versions](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/gurobi-machinelearning)](https://pypi.org/project/gurobi-machinelearning)
[![ReadTheDocs](https://readthedocs.com/projects/gurobi-optimization-gurobi-machine-learning/badge/?version=stable)](https://gurobi-optimization-gurobi-machine-learning.readthedocs-hosted.com)
[![Gurobi-forum](https://img.shields.io/badge/Help-Gurobi--Forum-red)](https://support.gurobi.com/hc/en-us/community/topics/10373864542609-GitHub-Projects)

[![Gurobi](https://raw.githubusercontent.com/Gurobi/gurobi-machinelearning/main/docs/source/_static/gurobi_light.png)](https://www.gurobi.com)


# Gurobi Machine Learning

Gurobi Machine Learning is an [open-source](https://gurobi-machinelearning.readthedocs.io/en/latest/meta/license.html) python package to formulate trained regression models in a [`gurobipy`](https://pypi.org/project/gurobipy/) model to be solved with the Gurobi solver.

The package currently supports various [scikit-learn](https://scikit-learn.org/stable/) objects. It has limited support for the [Keras](https://keras.io/) API of [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/) and [XGBoost](https://www.xgboost.ai). Only neural networks with ReLU activation can be used with Keras and PyTorch.

# Documentation

The latest user manual is available on [readthedocs](https://gurobi-machinelearning.readthedocs.io/).

# Contact us

For questions related to using Gurobi Machine Learning please use [Gurobi's Forum](https://support.gurobi.com/hc/en-us/community/topics/10373864542609-GitHub-Projects).

For reporting bugs, issues and feature requests please
[open an issue](https://github.com/Gurobi/gurobi-machinelearning/issues).

If you encounter issues with Gurobi or ``gurobipy`` please contact
[Gurobi Support](https://support.gurobi.com/hc/en-us).

# Installation

## Dependencies

`gurobi-machinelearning` requires the following:
- Python >= 3.9
- [`numpy`](https://pypi.org/project/numpy/) >= 1.23.0
- [`gurobipy`](https://pypi.org/project/gurobipy/) >= 10.0
- [`scipy`](https://pypi.org/project/scipy/) >= 1.9.3

The current version supports the following ML packages:
- [`torch`](https://pypi.org/project/torch/)
- [`scikit-learn`](https://pypi.org/project/scikit-learn)
- [`tensorflow`](https://pypi.org/project/tensorflow)
- [`XGBoost`](https://pypi.org/project/xgboost/)

Installing these packages is only required if the predictor you want to insert uses them
(i.e. to insert a Keras based predictor you need to have `tensorflow` installed).

The up to date supported and tested versions of each package for the last release can be
[found in the documentation](https://gurobi-machinelearning.readthedocs.io/en/stable/user/start.html#id7).

## Pip installation

The easiest way to install `gurobi-machinelearning` is using `pip` in a virtual environment:
```shell
(.venv) pip install gurobi-machinelearning
```
This will also install the `numpy`, `scipy` and `gurobipy` dependencies.

Please note that `gurobipy` is commercial software and requires a license. When installed via pip or conda,
`gurobipy` ships with a free license which is only for testing and can only solve models of limited size.

# Getting a Gurobi License
Alternatively to the bundled limited license, there are licenses that can handle models of all sizes.

As a student or staff member of an academic institution you qualify for a free, full product license.
For more information, see:

* https://www.gurobi.com/academia/academic-program-and-licenses/

For a commercial evaluation, you can
[request an evaluation license](https://www.gurobi.com/free-trial/?utm_source=internal&utm_medium=documentation&utm_campaign=fy21_pipinstall_eval_pypipointer&utm_content=c_na&utm_term=pypi).

Other useful resources to get started:
* https://www.gurobi.com/documentation/
* https://support.gurobi.com/hc/en-us/community/topics/

# Development
We value any level of experience in using Gurobi Machine Learning and would like to encourage you to
contribute directly to this project. Please see the [Contributing Guide](CONTRIBUTING.md) for more information.

## Source code
You can clone the latest sources with the command:
```shell
git clone git@github.com:Gurobi/gurobi-machinelearning.git
```

## Testing
After cloning the project, you can run the tests by invoking `tox`. For this, you will need to create a virtual
environment and activate it:
```shell
python3.10 -m venv .venv
. .venv/bin/activate
```
Then, you can install `tox` (>= 3.26.0) and run a few basic tests:
```shell
(.venv) pip install tox
(.venv) tox -e py310,pre-commit,docs
```
`tox` will install, among others, the aforementioned ML packages into a separate `venv`. These packages can be quite
large, so this might take a while.

### Running the full test set
In the above command, we only ran a subset of tests. Running the full set of tests requires having a Gurobi license
installed, and is done by running just the `tox` command without the `-e` parameter:

```shell
(.venv) pip install tox
(.venv) tox
```

If you don't have a Gurobi license, you can still run the subset of tests, open a PR, and Github Actions will run the
tests with a full Gurobi license.

## Submitting a Pull Request
Before opening a Pull Request, have a look at the full [Contributing page](CONTRIBUTING.md) to make sure your code
complies with our guidelines.
