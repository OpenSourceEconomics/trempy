#!/usr/bin/env python
"""This module updates and checks the regression vault."""
import subprocess
import shutil
import sys

from trempy.config_trempy import PACKAGE_DIR

PYTHON_EXEC = sys.executable
NUM_TESTS = 10000


# We create a new regression vault.
cmd = PYTHON_EXEC + ' run.py --request create --tests ' + str(NUM_TESTS)
subprocess.check_call(cmd, shell=True)

# We integrate it in the package.
shutil.copy('regression_vault.trempy.pkl', PACKAGE_DIR + '/tests')

# We immediately check the new vault. Problems can arise here if the randomness is not properly
# controlled for.
cmd = PYTHON_EXEC + ' run.py --request check --tests ' + str(NUM_TESTS)
subprocess.check_call(cmd, shell=True)
