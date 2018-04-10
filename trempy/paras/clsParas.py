"""This module contains the class for the collection of parameters."""
import numpy as np

from trempy.custom_exceptions import TrempyError
#from interalpy.logging.clsLogger import logger_obj
#from interalpy.config_interalpy import SMALL_FLOAT
from trempy.config_trempy import HUGE_FLOAT
#from interalpy.config_interalpy import NUM_PARAS
from trempy.shared.clsBase import BaseCls
from trempy.paras.clsPara import ParaCls


class ParasCls(BaseCls):
    """This class manages all issues about the model specification."""
    def __init__(self, init_dict):
        """This method initializes the parameter class."""

        self.attr = dict()

        self.attr['para_objs'] = []

        self.attr['para_labels'] = []
        # preference parameters
        for label in ['alpha', 'eta', 'beta']:
            value, is_fixed, bounds = init_dict['PREFERENCES'][label]
            self.attr['para_objs'] += [ParaCls(label, value, is_fixed, bounds)]
            self.attr['para_labels'] += [label]

        for label in init_dict['QUESTIONS'].keys():
            value, cutoffs = init_dict['QUESTIONS'][label]
            self.attr['para_objs'] += [ParaCls(label, value, False, cutoffs)]
            self.attr['para_labels'] += [label]

        self.check_integrity()

    def get_para(self, label):
        """This method allows to access a single parameter."""
        # Distribute class attributes
        para_objs = self.attr['para_objs']

        for para_obj in para_objs:
            if label == para_obj.get_attr('label'):
                rslt = list()
                for info in ['value', 'is_fixed', 'bounds']:
                    rslt += [para_obj.get_attr(info)]
                    print(rslt)
                return rslt

        raise TrempyError('parameter not available')

    def set_values(self, perspective, which, values):
        """This method allows to directly set the values of the parameters."""
        # Distribute class attributes
        para_objs = self.attr['para_objs']

        count = 0

        for label in self.attr['para_labels']:
            for para_obj in para_objs:
                # We are only interested in the free parameters.
                if which == 'free' and para_obj.get_attr('is_fixed'):
                    continue
                # We are only interested in one particular parameter.
                if label != para_obj.get_attr('label'):
                    continue

                if perspective in ['econ']:
                    value = values[count]
                elif perspective in ['optim']:
                    bounds = para_obj.get_attr('bounds')
                    value = self._to_econ(values[count], bounds)
                else:
                    raise TrempyError('misspecified request')

                para_obj.set_attr('value', value)

                para_obj.check_integrity()

                count += 1

    def get_values(self, perspective, which):
        """This method allow to directly access the values of the parameters."""
        # Distribute class attributes
        para_objs = self.attr['para_objs']

        # Initialize containers
        values = list()

        for label in self.attr['para_labels']:
            for para_obj in para_objs:
                # We are only interested in the free parameters.
                if which == 'free' and para_obj.get_attr('is_fixed'):
                    continue
                # We are only interested in one particular parameter.
                if label != para_obj.get_attr('label'):
                    continue

                if perspective in ['econ']:
                    value = para_obj.get_attr('value')
                elif perspective in ['optim']:
                    value = self._to_optimizer(para_obj)
                else:
                    raise InteralpyError('misspecified request')

                values += [value]

        return values

    def check_integrity(self):
        """This method checks some basic features of the class that need to hold true at all
        times."""
        # Distribute class attributes
        para_objs = self.attr['para_objs']

        for para_obj in para_objs:
            para_obj.check_integrity()

    def _to_optimizer(self, para_obj):
        """This method transfers a single parameter to its value used by the optimizer."""
        # Distribute attributes
        lower, upper = para_obj.get_attr('bounds')
        value = self._to_real(para_obj.get_attr('value'), lower, upper)
        return value

    def _to_econ(self, value, bounds):
        """This function transforms parameters over the whole real to a bounded interval."""
        value = self._to_interval(value, *bounds)
        return value

    @staticmethod
    def _to_interval(val, lower, upper):
        """This function maps any value to a bounded interval."""
        try:
            exponential = np.exp(-val)
        except (OverflowError, FloatingPointError) as _:
            exponential = HUGE_FLOAT
            logger_obj.record_event(2)
        interval = upper - lower
        return lower + interval / (1 + exponential)

    @staticmethod
    def _to_real(value, lower, upper):
        """This function transforms the bounded parameter back to the real line."""
        if np.isclose(value, lower):
            value += SMALL_FLOAT
            logger_obj.record_event(1)
        elif np.isclose(value, upper):
            value -= SMALL_FLOAT
            logger_obj.record_event(1)
        else:
            pass

        interval = upper - lower
        transform = (value - lower) / interval
        return np.log(transform / (1.0 - transform))
