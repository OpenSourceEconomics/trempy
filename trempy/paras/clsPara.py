"""This module contains the class for a single parameter."""
import numpy as np

from trempy.config_trempy import SMALL_FLOAT
from trempy.shared.clsBase import BaseCls


class ParaCls(BaseCls):
    """This class manages all issues about a single parameter."""
    def __init__(self, label, value, is_fixed, bounds):
        """This method initializes the parameter class."""
        self.attr = dict()

        self.attr['is_fixed'] = is_fixed
        self.attr['bounds'] = bounds
        self.attr['label'] = label
        self.attr['value'] = value

        self.check_integrity()

    def check_integrity(self):
        """This method checks the integrity of the parameter."""
        # Distribute class attributes
        lower, upper = self.attr['bounds']
        is_fixed = self.attr['is_fixed']
        value = self.attr['value']

        # Check several conditions that need to hold true at all times.
        cond = is_fixed in [True, False]
        np.testing.assert_equal(cond, True)

        # Check whether the parameters are within their specified bounds.
        cond = lower - SMALL_FLOAT <= value <= upper + SMALL_FLOAT
        if cond is False:
            print('flaw')
        np.testing.assert_equal(cond, True)
