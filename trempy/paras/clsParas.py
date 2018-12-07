"""This module contains the class for the collection of parameters."""
import numpy as np

from trempy.config_trempy import PREFERENCE_PARAMETERS
from trempy.custom_exceptions import TrempyError
from trempy.record.clsLogger import logger_obj
from trempy.config_trempy import SMALL_FLOAT
from trempy.config_trempy import HUGE_FLOAT
from trempy.shared.clsBase import BaseCls
from trempy.paras.clsPara import ParaCls


class ParasCls(BaseCls):
    """Manage all issues about the model specification."""

    def __init__(self, init_dict):
        """Initialize the parameter class."""
        version = init_dict['VERSION']['version']

        self.attr = dict()
        self.attr['optimizer'] = init_dict['ESTIMATION']['optimizer']
        self.attr['version'] = version
        self.attr['para_labels'] = []
        self.attr['para_objs'] = []

        if version in ['nonstationary']:
            discounting = init_dict['VERSION']['discounting']
            stationary_model = init_dict['VERSION']['stationary_model']
            self.attr['discounting'] = discounting
            self.attr['stationary_model'] = stationary_model

        # Preference parameters are handled for each version separately.
        for label in PREFERENCE_PARAMETERS[version]:

            if version in ['scaled_archimedean']:
                if label in ['r_self']:
                    value, is_fixed, bounds = init_dict['UNIATTRIBUTE SELF']['r']
                elif label in ['r_other']:
                    value, is_fixed, bounds = init_dict['UNIATTRIBUTE OTHER']['r']
                else:
                    value, is_fixed, bounds = init_dict['MULTIATTRIBUTE COPULA'][label]

            elif version in ['nonstationary']:
                if label in ['alpha', 'beta', 'gamma', 'y_scale']:
                    value, is_fixed, bounds = init_dict['ATEMPORAL'][label]
                elif (label.startswith('discount_factors') or
                      label.startswith('unrestricted_weights')):
                    value, is_fixed, bounds = init_dict['DISCOUNTING'][label]
                else:
                    raise TrempyError('parameter label not implemented')

            else:
                raise TrempyError('version not implemented')

            self.attr['para_objs'] += [ParaCls(label, value, is_fixed, bounds)]
            self.attr['para_labels'] += [label]

        # Record created parameters so we can use that later in estimate step to get
        #  standard deviations without using hard-coded numbers
        self.attr['nparas_econ'] = len(self.attr['para_objs'])

        # QUESTION specific parameters
        for label in sorted(init_dict['QUESTIONS'].keys()):
            value, is_fixed, bounds = init_dict['QUESTIONS'][label]
            self.attr['para_objs'] += [ParaCls(int(label), value, is_fixed, bounds)]
            self.attr['para_labels'] += [int(label)]

        self.attr['nparas_questions'] = len(self.attr['para_objs']) - self.attr['nparas_econ']

        self.check_integrity()

    def get_para(self, label):
        """Access a single parameter and get value, free/fixed and bounds."""
        # Distribute class attributes
        para_objs = self.attr['para_objs']

        for para_obj in para_objs:
            if label == para_obj.get_attr('label'):
                rslt = [para_obj.get_attr(info) for info in ['value', 'is_fixed', 'bounds']]
                return rslt

        raise TrempyError('parameter not available')

    def set_values(self, perspective, which, values):
        """Directly set the values of the parameters."""
        # Antibugging
        np.testing.assert_equal(which in ['all', 'free'], True)

        # Distribute class attributes
        para_objs = self.attr['para_objs']
        optimizer = self.attr['optimizer']

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
                    value = self._to_econ(values[count], bounds, optimizer)
                else:
                    raise TrempyError('misspecified request')

                para_obj.set_attr('value', value)
                para_obj.check_integrity()
                count += 1

    def get_values(self, perspective, which):
        """Directly access the values of the parameters."""
        # Antibugging
        np.testing.assert_equal(which in ['all', 'free'], True)

        # Distribute class attributes
        para_objs = self.attr['para_objs']
        optimizer = self.attr['optimizer']

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
                    # Handle choice of algorithm
                    value = self._to_optimizer(para_obj, optimizer)
                else:
                    raise TrempyError('misspecified request')

                values += [value]
        return values

    def get_bounds(self, which):
        """Directly return a list of bounds for the parameters."""
        # Antibugging
        np.testing.assert_equal(which in ['all', 'free'], True)

        # Distribute class attributes
        para_objs = self.attr['para_objs']

        bounds = list()

        for label in self.attr['para_labels']:
            for para_obj in para_objs:
                # We are only interested in the free parameters.
                if which == 'free' and para_obj.get_attr('is_fixed'):
                    continue
                # We are only interested in one particular parameter.
                if label != para_obj.get_attr('label'):
                    continue
                lower, upper = para_obj.get_attr('bounds')
                bounds += [(lower, upper)]
        return bounds

    def check_integrity(self):
        """Check some basic features of the class that need to hold true at all times."""
        para_objs = self.attr['para_objs']

        for para_obj in para_objs:
            para_obj.check_integrity()

    def _to_optimizer(self, para_obj, optimizer):
        """Transfer a single parameter to its value used by the optimizer."""
        lower, upper = para_obj.get_attr('bounds')
        # Optimizer that support bounds.
        if optimizer == 'SCIPY-L-BFGS-B':
            value = para_obj.get_attr('value')
        else:
            # Optimizer that do not support bounds.
            value = self._to_real(para_obj.get_attr('value'), lower, upper)
        return value

    def _to_econ(self, value, bounds, optimizer):
        """Transform parameters over the whole real to a bounded interval."""
        if optimizer == 'SCIPY-L-BFGS-B':
            lower, upper = bounds
            if np.isclose(value, lower):
                value += SMALL_FLOAT
                logger_obj.record_event(0)
            elif np.isclose(value, upper):
                value -= SMALL_FLOAT
                logger_obj.record_event(0)
            else:
                pass
            return value
        # Optimizer without support for bounds need to convert back from real to interval.
        value = self._to_interval(value, *bounds)
        return value

    @staticmethod
    def _to_interval(val, lower, upper):
        """Map any value to a bounded interval."""
        # Handle optional arguments
        if val is None:
            return None

        try:
            exponential = np.exp(-val)
        except (OverflowError, FloatingPointError):
            exponential = HUGE_FLOAT
            logger_obj.record_event(1)
        interval = upper - lower
        return lower + interval / (1 + exponential)

    @staticmethod
    def _to_real(value, lower, upper):
        """Transform the bounded parameter back to the real line."""
        # Handle optional arguments with None value.
        if value is None:
            return None

        if np.isclose(value, lower):
            value += SMALL_FLOAT
            logger_obj.record_event(0)
        elif np.isclose(value, upper):
            value -= SMALL_FLOAT
            logger_obj.record_event(0)
        else:
            pass

        interval = upper - lower
        transform = (value - lower) / interval
        return np.log(transform / (1.0 - transform))
