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

from importlib import metadata

try:
    __version__ = metadata.version("gurobi_machinelearning")
except metadata.PackageNotFoundError:
    __version__ = "dev"

GIT_HASH = "$Format:%H$"


def get_versions():
    # Downloaded package with inserted git hash.
    if "Format" not in GIT_HASH:
        git_hash = f"-{GIT_HASH}"
    # No inserted git hash, the repo is probably cloned.
    else:
        git_hash = ""

    return {"short": __version__, "long": f"{__version__}{git_hash}"}
