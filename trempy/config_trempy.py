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
QUESTIONS_ALL = [13] + list(range(31, 46))

# We want to be strict about any problems due to floating-point errors. However, during estimation,
# we might have a problem with UNDERFLOW when evaluating the probability density function.
np.seterr(divide='raise', over='raise', invalid='raise', under='ignore')

# We need to impose some bounds on selected estimation parameters. The bounds are included in the
# package's admissible values.
DEFAULT_BOUNDS = dict()
DEFAULT_BOUNDS['r_other'] = [-5.00, 5.00]
DEFAULT_BOUNDS['r_self'] = [-5.00, 5.00]
DEFAULT_BOUNDS['delta'] = [0.01, 1.00]
DEFAULT_BOUNDS['other'] = [0.00, 0.99]
DEFAULT_BOUNDS['self'] = [0.00, 0.99]

for q in QUESTIONS_ALL:
    DEFAULT_BOUNDS[q] = [0.01, 100]

# We maintain a list of all preference parameters.
PREFERENCE_PARAMETERS = ['r_self', 'r_other', 'delta', 'self', 'other']
