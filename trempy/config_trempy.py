"""This module contains some configuration information."""
import sys
import os

import numpy as np

# We only support Python 3.
# TODO: This needs to be checked properly.
#np.testing.assert_equal(sys.executable[0], '3')

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_RESOURCES_DIR = PACKAGE_DIR + '/tests'

SMALL_FLOAT = 10e-10
HUGE_FLOAT = 10e+20
TINY_FLOAT = 10e-20

# We want to be strict about any problems due to floating-point errors.
np.seterr(all='raise')

# We need to impose some bounds on selected estimation parameters. The bounds are included in the
# package's admissible values.
DEFAULT_BOUNDS = dict()
DEFAULT_BOUNDS['alpha'] = [0.01, 0.99]
DEFAULT_BOUNDS['beta'] = [0.01, 5.00]
DEFAULT_BOUNDS['eta'] = [0.01, 0.99]
