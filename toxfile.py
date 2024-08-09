import shutil
from tox.plugin import impl
from tox.tox_env.api import ToxEnv

@impl
def tox_env_teardown(tox_env: ToxEnv):
    print(f"removing env dir: {tox_env}")
    shutil.rmtree(tox_env.env_dir)
