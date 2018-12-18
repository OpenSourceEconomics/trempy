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
QUESTIONS_ALL = list(range(1, 46))

# We want to be strict about any problems due to floating-point errors. However, during estimation,
# we might have a problem with UNDERFLOW when evaluating the probability density function.
np.seterr(divide='raise', over='raise', invalid='raise', under='ignore')

# We need to impose some bounds on selected estimation parameters. The bounds are included in the
# package's admissible values.

# Scaled Archimedean copula parameter
DEFAULT_BOUNDS = dict()
DEFAULT_BOUNDS['r_other'] = [0.01, 4.99]
DEFAULT_BOUNDS['r_self'] = [0.01, 4.99]
DEFAULT_BOUNDS['delta'] = [0.01, 4.99]
DEFAULT_BOUNDS['other'] = [0.00, 0.99]
DEFAULT_BOUNDS['self'] = [0.00, 0.99]

# Nonstationary utility function
DEFAULT_BOUNDS['alpha'] = [0.01, 4.99]
DEFAULT_BOUNDS['beta'] = [0.01, 4.99]
DEFAULT_BOUNDS['gamma'] = [0.01, 4.99]
DEFAULT_BOUNDS['y_scale'] = [0.01, 14.99]

DEFAULT_BOUNDS['discount_factors_0'] = [0.01, 1.001]
DEFAULT_BOUNDS['discount_factors_1'] = [0.01, 1.001]
DEFAULT_BOUNDS['discount_factors_3'] = [0.01, 1.001]
DEFAULT_BOUNDS['discount_factors_6'] = [0.01, 1.001]
DEFAULT_BOUNDS['discount_factors_12'] = [0.01, 1.001]
DEFAULT_BOUNDS['discount_factors_24'] = [0.01, 1.001]

DEFAULT_BOUNDS['unrestricted_weights_0'] = [0.01, 1.001]
DEFAULT_BOUNDS['unrestricted_weights_1'] = [0.01, 1.001]
DEFAULT_BOUNDS['unrestricted_weights_3'] = [0.01, 1.001]
DEFAULT_BOUNDS['unrestricted_weights_6'] = [0.01, 1.001]
DEFAULT_BOUNDS['unrestricted_weights_12'] = [0.01, 1.001]
DEFAULT_BOUNDS['unrestricted_weights_24'] = [0.01, 1.001]

for q in QUESTIONS_ALL:
    DEFAULT_BOUNDS[q] = [0.01, 100.00]

# We maintain a list of all preference parameters by version.
PREFERENCE_PARAMETERS = {
    'scaled_archimedean': ['r_self', 'r_other', 'delta', 'self', 'other'],
    'nonstationary': ['alpha', 'beta', 'gamma', 'y_scale',
                      'discount_factors_0', 'discount_factors_1',
                      'discount_factors_3', 'discount_factors_6',
                      'discount_factors_12', 'discount_factors_24',
                      'unrestricted_weights_0', 'unrestricted_weights_1',
                      'unrestricted_weights_3', 'unrestricted_weights_6',
                      'unrestricted_weights_12', 'unrestricted_weights_24'],
}

# We need to specify the grid for the determination of the optimal compensation.
# It varies as the payoff turns negative at different values.
LOTTERY_BOUNDS = {
    # TEMPORAL CHOICES

    # Univariate discounting: SELF. 0-1, 0-3, 0-6, 0-12, 0-24, 6-12
    1: [+50.0, 56.25],
    2: [+50.0, 68.75],
    3: [+50.0, 87.50],
    4: [+50.0, 125.00],
    5: [+50.0, 200.00],
    6: [+50.0, 87.50],

    # Univariate discounting: CHARITY. 0-1, 0-3, 0-6, 0-12, 0-24, 6-12
    7: [+50.0, 56.25],
    8: [+50.0, 68.75],
    9: [+50.0, 87.50],
    10: [+50.0, 125.00],
    11: [+50.0, 200.00],
    12: [+50.0, 87.50],

    # Exchange rate. 0-0, 1-1, 3-3, 6-6, 12-12, 24-24
    # Question 13 is counted as a riskless lottery question because t=0.
    13: [+0.00, 200.00],
    14: [+0.00, 200.00],
    15: [+0.00, 200.00],
    16: [+0.00, 200.00],
    17: [+0.00, 200.00],
    18: [+0.00, 200.00],

    # Multivariate discounting: SELF. 0-1, 0-3, 0-6, 0-12, 0-24, 6-12
    19: [+0.00, 168.75],
    20: [+0.00, 206.25],
    21: [+0.00, 262.50],
    22: [+0.00, 375.00],
    23: [+0.00, 600.00],
    24: [+0.00, 262.50],

    # Multivariate discounting: CHARITY. 0-1, 0-3, 0-6, 0-12, 0-24, 6-12
    25: [+0.00, 56.25],
    26: [+0.00, 68.75],
    27: [+0.00, 87.50],
    28: [+0.00, 125.00],
    29: [+0.00, 200.00],
    30: [+0.00, 87.50],

    # RISKY CHOICES
    31: [-5.0000, 5.0000],
    32: [-10.0000, 10.0000],
    33: [-20.0000, 20.0000],
    34: [-5.0000, 5.0000],
    35: [-10.0000, 10.0000],
    36: [-20.0000, 20.0000],
    37: [-5.0000, 5.0000],
    38: [-10.0000, 10.0000],
    39: [-20.0000, 20.0000],
    40: [-5.0000, 5.0000],
    41: [-5.0000, 5.0000],
    42: [-2.0000, 5.0000],
    43: [-5.0000, 5.0000],
    44: [-5.0000, 5.0000],
    45: [-2.0000, 5.0000],
}
