[![build and test](https://github.com/Gurobi/gurobi-machinelearning/actions/workflows/push.yml/badge.svg)](https://github.com/Gurobi/gurobi-machinelearning/actions/workflows/push.yml)
![Python versions](https://img.shields.io/badge/python-3.9%20|%203.10-blue)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/gurobipy)](https://pypi.org/project/gurobipy)

![Gurobi](doc_source/source/_static/image8.png)


> **⚠ Warning**
> ```This code is in a pre-release state. It may not be fully functional and breaking changes can occur without notice.```

# Gurobi Machine Learning

Gurobi Machine Learning is a python package to insert trained predictors into a [`gurobipy`](https://pypi.org/project/gurobipy/) model.

The goal of the package is to:
  1. Simplify the process of importing a trained machine learning model built with a popular ML package into an optimization model.
  1. Improve algorithmic performance to enable the optimization model to explore a sizable space of solutions that satisfy the variable relationships captured in the ML model.
  1. Make it easier for optimization models to mix explicit and implicit constraints.

The package currently supports the following regression models:
 - [Scikit-learn](https://scikit-learn.org/)
    - Regression models
      - Linear regression
      - Logistic regression
      - Neural-network regression (`MLPRegressor`)
      - Decision tree
      - Gradient boosting tree
      - Random Forest
    - Transformers
      - Standard Scaler
      - Polynomial Features (degree 2)
 - [Keras](https://keras.io/)
   - Dense layers
   - ReLU layers
 - [PyTorch](https://pytorch.org/) (only `torch.nn.Sequential` objects)
   - Dense layers
   - ReLU layers

Our documentation contains more detailed information on the
[supported models](https://gurobi-machinelearning.readthedocs.io/en/stable/).

# Installation

## Dependencies

`gurobi-machinelearning` requires the following:
- Python >= 3.9
- [`numpy`](https://pypi.org/project/numpy/) >= 1.22.0
- [`gurobipy`](https://pypi.org/project/gurobipy/) >= 10.0

The current version supports the following ML package versions:
- [`torch`](https://pypi.org/project/torch/1.12.1/) == 1.12.1
- [`scikit-learn`](https://pypi.org/project/scikit-learn/1.1.2/) == 1.1.2
- [`tensorflow`](https://pypi.org/project/tensorflow/2.10.0/) == 2.10.0

Installing these package is only required if the predictor you want to insert uses them
(i.e. to insert a Keras based predictor you need to have `tensorflow` installed).

## Pip installation

The easiest way to install `gurobi-machinelearning` is using `pip` in a virtual environment:
```shell
(.venv) pip install gurobi-machinelearning
```
This will also install the `numpy` and `gurobipy` dependencies.

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
