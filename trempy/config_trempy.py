"""This module contains some configuration information."""
import numpy as np

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
