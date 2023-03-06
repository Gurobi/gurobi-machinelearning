# Contributing to Gurobi Machine Learning

Welcome to Gurobi Machine Learning!

We value your experience in using Gurobi Machine Learning and would like to encourage you to
contribute directly to this project.

## How to report bugs or submit feature requests
If you encounter a bug, or you think there is a need for a new feature, we recommend to
first add the bug report or feature request to the Gurobi Machine Learning's [GitHub issue
tracker](https://github.com/Gurobi/gurobi-machinelearning/issues).

It would be great if you add a minimal reproducible example when reporting a bug, or
include reasoning on how the new requested feature improves the code.

## Submitting changes
We welcome external contributions to Gurobi Machine Learning.
Note that all contributors should accept the [Contributor License Agreement (CLA)](https://gist.github.com/mattmilten/d1c9640d79bde0ece8c2f46152639011).

To contribute code you should use the GitHub pull request workflow. Once your pull request is ready for review, one
of the core maintainers of Gurobi Machine Learning will review your pull request.

A pull request should contain tests for the changes made to the code behavior, should
include a clear message outlining the changes done, and should be linked to an existing
issue.

Before submitting a pull request:
- install the [pre-commit](https://pre-commit.com) package to enable the automatic
  running of the pre-commit hooks in the `.pre-commit-configuration.yaml` file,
- make sure all tests pass by running `tox` in the root folder of the `gurobi-machinelearning`.
- add any other relevant checks for your changes to Gurobi Machine Learning.

After a pull request is submitted, tests will be run, and the status
will appear on the pull request page. If the tests failed, there is a link which can be used to debug the failed tests.

## Code reviews
The pull request author should respond to all comments received. If the
comment has been accepted and appropriate changes applied, the author should respond by
a short message such as "Done" and then resolve the comment. If more discussion is
needed on a comment, it should remain open until a solution can be figured out.

## Merging changes
The core maintainer that reviewed the pull request will merge it after all comments have been addressed and when all tests are passing.
