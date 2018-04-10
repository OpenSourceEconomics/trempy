"""This module contains some configuration information."""
import os

import numpy as np

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
#TEST_RESOURCES_DIR = PACKAGE_DIR + '/tests/material'

SMALL_FLOAT = 10e-10
HUGE_FLOAT = 10e+20
TINY_FLOAT = 10e-20

# We are strict in enforcing data types throughout.
DATA_DTYPES = dict()
DATA_DTYPES.update({'Participant.code': str, 'Question': np.int, 'x': np.float,  'y': np.float})
DATA_DTYPES.update({'I1': np.float, 'I2': np.float, 'm': np.float, 'D': np.int})

# We want to be strict about any problems due to floating-point errors.
np.seterr(all='raise')

# We need to impose some bounds on selected estimation parameters. The bounds are included in the
# package's admissible values.
DEFAULT_BOUNDS = dict()
DEFAULT_BOUNDS['alpha'] = [0.01, 0.99]
DEFAULT_BOUNDS['beta'] = [0.01, 5.00]
DEFAULT_BOUNDS['eta'] = [0.01, 0.99]
