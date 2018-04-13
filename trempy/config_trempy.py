"""This module contains some configuration information."""
import sys
import os

import numpy as np

# We only support Python 3.
np.testing.assert_equal(sys.version_info[0], 3)

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

SMALL_FLOAT = 10e-10
HUGE_FLOAT = 10e+20
TINY_FLOAT = 10e-20

NEVER_SWITCHERS = 9999

# We set the range of questions that are possible to handle.
QUESTIONS_ALL = range(1, 16)

# We want to be strict about any problems due to floating-point errors.
np.seterr(all='raise')

# We need to impose some bounds on selected estimation parameters. The bounds are included in the
# package's admissible values. We need to make sure that zero is an admissible value.
DEFAULT_BOUNDS = dict()
DEFAULT_BOUNDS['alpha'] = [0.00 - 2 * SMALL_FLOAT, 0.99]
DEFAULT_BOUNDS['beta'] = [0.00 - 2 * SMALL_FLOAT, 0.99]
DEFAULT_BOUNDS['eta'] = [0.00 - 2 * SMALL_FLOAT, 0.99]

for q in QUESTIONS_ALL:
    DEFAULT_BOUNDS[q] = [0.01, 100]
