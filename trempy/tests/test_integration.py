"""This module contains some integration tests."""
from subprocess import CalledProcessError
import subprocess
import os

import numpy as np

from trempy.tests.test_auxiliary import get_random_init
from trempy.config_trempy import PACKAGE_DIR
from trempy import simulate
from trempy import estimate


def test_1():
    """This test simply runs the core workflow of simulation and estimation."""
    constr = dict()
    constr['maxfun'] = np.random.random_integers(1, 5)

    get_random_init(constr)
    simulate('test.trempy.ini')
    estimate('test.trempy.ini')


def test_2():
    """This test runs flake8 to ensure the code quality. However, this is only relevant during
    development."""
    try:
        import flake8    # noqa: F401
    except ImportError:
        return None

    cwd = os.getcwd()
    os.chdir(PACKAGE_DIR)
    try:
        subprocess.check_call(['flake8'])
        os.chdir(cwd)
    except CalledProcessError:
        os.chdir(cwd)
        raise CalledProcessError
