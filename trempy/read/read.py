"""This module contains all the required capabilities to read an initialization file."""
import shlex
import os

import numpy as np

from trempy.custom_exceptions import TrempyError
from trempy.config_trempy import DEFAULT_BOUNDS
from trempy.config_trempy import QUESTIONS_ALL
from trempy.config_trempy import HUGE_FLOAT

# Blocks that should be processed all the time.
BASIC_GROUPS = [
    'VERSION', 'SIMULATION', 'ESTIMATION', 'SCIPY-BFGS', 'SCIPY-POWELL', 'SCIPY-L-BFGS-B',
    'CUTOFFS', 'QUESTIONS',
]

# Blocks that are specific to the 'version' of the utility function.
ESTIMATION_GROUP = {
    'scaled_archimedean': ['UNIATTRIBUTE SELF', 'UNIATTRIBUTE OTHER', 'MULTIATTRIBUTE COPULA'],
    'nonstationary': ['ATEMPORAL', 'DISCOUNTING'],
}


def read(fname):
    """Read the initialization file."""
    # Check input
    np.testing.assert_equal(os.path.exists(fname), True)

    # Initialization
    dict_, group = {}, None

    with open(fname) as in_file:

        for line in in_file.readlines():

            list_ = shlex.split(line)

            # Determine special cases
            is_empty, is_group, is_comment = process_cases(list_)

            # Applicability
            if is_empty or is_comment:
                continue

            # Prepare dictionary
            if is_group:
                group = ' '.join(list_)
                dict_[group] = dict()
                continue

            # Code below is only executed if the current line is not a group name
            flag, value = list_[:2]

            # Handle the VERSION block.
            if (group in ['VERSION']) and (flag in ['version']):
                version = value

            # Type conversions for the NON-CUTOFF block
            if group not in ['CUTOFFS']:
                value = type_conversions(version, flag, value)

            # We need to make sure questions and cutoffs are not duplicated.
            if flag in dict_[group].keys():
                raise TrempyError('duplicated information')

            # Handle the basic blocks
            if group in BASIC_GROUPS:
                if group in ['CUTOFFS']:
                    dict_[group][flag] = process_cutoff_line(list_)
                elif group in ['QUESTIONS']:
                    dict_[group][flag] = process_coefficient_line(group, list_, value)
                else:
                    dict_[group][flag] = value

            # Handle blocks specific to the 'version' of the utility function.
            if group in ESTIMATION_GROUP[version]:
                if version in ['scaled_archimedean']:
                    if flag not in ['max', 'marginal']:
                        dict_[group][flag] = process_coefficient_line(group, list_, value)
                    else:
                        dict_[group][flag] = value

                elif version in ['nonstationary']:
                    dict_[group][flag] = process_coefficient_line(group, list_, value)

                else:
                    raise TrempyError('version not implemented')

    # We allow for initialization files where no CUTOFFS are specified.
    if "CUTOFFS" not in dict_.keys():
        dict_['CUTOFFS'] = dict()

    # We want to ensure that the keys to the questions are integers
    for label in ['QUESTIONS', 'CUTOFFS']:
        dict_[label] = {int(x): dict_[label][x] for x in dict_[label].keys()}

    # We do some modifications on the cutoff values. Instead of None, we will simply use
    # HUGE_FLOAT and we fill up any missing cutoff values for any possible questions..
    for q in QUESTIONS_ALL:
        if q not in dict_['CUTOFFS'].keys():
            dict_['CUTOFFS'][q] = [-HUGE_FLOAT, HUGE_FLOAT]
        else:
            for i in range(2):
                if dict_['CUTOFFS'][q][i] is None:
                    dict_['CUTOFFS'][q][i] = (-1)**i * -HUGE_FLOAT

    # Enforce input requirements for optional arguments
    check_optional_args(dict_)

    return dict_


def process_cutoff_line(list_):
    """Process a cutoff line."""
    cutoffs = []
    for i in [1, 2]:
        if list_[i] == 'None':
            cutoffs += [None]
        else:
            cutoffs += [float(list_[i])]

    return cutoffs


def process_bounds(bounds, label):
    """Extract the proper bounds."""
    bounds = bounds.replace(')', '')
    bounds = bounds.replace('(', '')
    bounds = bounds.split(',')
    for i in range(2):
        if bounds[i] == 'None':
            bounds[i] = float(DEFAULT_BOUNDS[label][i])
        else:
            bounds[i] = float(bounds[i])

    return bounds


def process_coefficient_line(group, list_, value):
    """Process a coefficient line and extracts the relevant information.

    We also impose the default values for the bounds here.
    """
    try:
        label = int(list_[0])
    except ValueError:
        label = list_[0]

    # We need to adjust the labels
    label_internal = label
    if label in ['r'] and 'SELF' in group:
        label_internal = 'r_self'
    elif label in ['r'] and 'OTHER' in group:
        label_internal = 'r_other'

    if len(list_) == 2:
        is_fixed, bounds = False, DEFAULT_BOUNDS[label_internal]
    elif len(list_) == 4:
        is_fixed = True
        bounds = process_bounds(list_[3], label_internal)
    elif len(list_) == 3:
        is_fixed = (list_[2] == '!')

        if not is_fixed:
            bounds = process_bounds(list_[2], label_internal)
        else:
            bounds = DEFAULT_BOUNDS[label_internal]

    return value, is_fixed, bounds


def process_cases(list_):
    """Process cases and determine whether group flag or empty line."""
    # Get information
    is_empty = (len(list_) == 0)

    if not is_empty:
        is_group = list_[0].isupper()
        is_comment = list_[0][0] == '#'
    else:
        is_group = False
        is_comment = False

    # Finishing
    return is_empty, is_group, is_comment


def type_conversions(version, flag, value):
    """Type conversions by version."""
    # Handle ESTIMATION, SIMULATION and VERSION
    if flag in ['seed', 'agents', 'maxfun', 'skip']:
        value = int(value)
    elif flag in ['version', 'file', 'optimizer', 'start']:
        value = str(value)
    elif flag in ['detailed']:
        assert (value.upper() in ['TRUE', 'FALSE'])
        value = (value.upper() == 'TRUE')
    # Handle SCIPY-BFGS, SCIPY-L-BFGS-B and SCIPY-POWELL
    elif flag in ['eps', 'gtol', 'ftol', 'xtol']:
        value = float(value)
    # Empty flags
    elif flag in []:
        value = value.upper()
    # Handle Scaled Archimedean
    elif flag in ['marginal']:
        value = str(value)
    elif flag in ['max']:
        value = int(value)
    elif flag in ['r', 'delta', 'other', 'self']:
        value = float(value)
    # Handle nonstationary
    elif flag in ['alpha', 'beta', 'gamma', 'y_scale'] or flag.startswith('discount_factors'):
        value = float(value)
    elif flag.startswith('unrestricted_weights_'):
        if value == 'None':
            value = None
        else:
            value = float(value)
    else:
        value = float(value)

    # TODO: add consistency check for unrestricted weights and
    # TODO: add additional tests for the init dict in separate file "read_init_check.py"

    # Finishing
    return value


def check_optional_args(init_dict):
    """Enforce input requirements for the init_dict."""
    version = init_dict['VERSION']['version']
    if version in ['scaled_archimedean']:
        pass
    elif version in ['nonstationary']:
        optional_args = ['unrestricted_weights_{}'.format(int(x)) for x in [0, 1, 3, 6, 12, 24]]
        for label in optional_args:
            # If optional argument is not used (None), then we fix it at None.
            # In this case, the optimizer is not confused!
            if label in init_dict['DISCOUNTING'].keys():
                value, is_fixed, bounds = init_dict['DISCOUNTING'][label]
                if value is None and is_fixed is False:
                    raise TrempyError('Optional argument misspecified.')
            else:
                raise TrempyError('Please set unused optional arguments to None in init file.')
