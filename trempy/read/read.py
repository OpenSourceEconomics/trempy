"""This module contains all the required capabilities to read an initialization file."""
import shlex
import os

import numpy as np

from trempy.config_trempy import DEFAULT_BOUNDS
from trempy.config_trempy import HUGE_FLOAT


def read(fname):
    """This function reads the initialization file."""
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
                group = list_[0]
                dict_[group] = {}
                continue

            flag, value = list_[:2]

            # Type conversions
            if group not in ['CUTOFFS']:
                value = type_conversions(flag, value)

            # We need to allow for additional information about the potential estimation
            # parameters.
            if group in ['PREFERENCES', 'QUESTIONS']:
                dict_[group][flag] = process_coefficient_line(list_, value)
            elif group in ['CUTOFFS']:
                dict_[group][flag] = process_cutoff_line(list_)
            else:
                dict_[group][flag] = value

    # We want to ensure that the keys to the questions are integers
    for label in ['QUESTIONS', 'CUTOFFS']:
        dict_[label] = {int(x): dict_[label][x] for x in dict_[label].keys()}

    # We do some modifications on the cutoff values. Instead of None, we will simply use
    # HUGE_FLOAT and we fill up any missing cutoff values.
    questions = dict_['QUESTIONS'].keys()
    for q in questions:
        if q not in dict_['CUTOFFS'].keys():
            dict_['CUTOFFS'][q] = [-HUGE_FLOAT, HUGE_FLOAT]
        else:
            for i in range(2):
                if dict_['CUTOFFS'][q][i] is None:
                    dict_['CUTOFFS'][q][i] = (-1)**i * -HUGE_FLOAT

    return dict_


def process_cutoff_line(list_):
    """This function processes a cutoff line."""
    cutoffs = []
    for i in [1, 2]:
        if list_[i] == 'None':
            cutoffs += [None]
        else:
            cutoffs += [float(list_[i])]

    return cutoffs


def process_bounds_cutoffs(bounds, label):
    """This function extracts the proper bounds."""
    bounds = bounds.replace(')', '')
    bounds = bounds.replace('(', '')
    bounds = bounds.split(',')
    for i in range(2):
        if bounds[i] == 'None':
            bounds[i] = DEFAULT_BOUNDS[label][i]
        else:
            bounds[i] = float(bounds[i])

    return bounds


def process_coefficient_line(list_, value):
    """This function processes a coefficient line and extracts the relevant information. We also
    impose the default values for the bounds here."""
    label = list_[0]

    if len(list_) == 2:
        is_fixed, bounds = False, DEFAULT_BOUNDS[label]
    elif len(list_) == 4:
        is_fixed = True
        bounds = process_bounds_cutoffs(list_[3], label)
    elif len(list_) == 3:
        is_fixed = (list_[2] == '!')

        if not is_fixed:
            bounds = process_bounds_cutoffs(list_[2], label)
        else:
            bounds = DEFAULT_BOUNDS[label]

    return value, is_fixed, bounds


def process_cases(list_):
    """Process cases and determine whether group flag or empty line."""
    # Get information
    is_empty = (len(list_) == 0)

    if not is_empty:
        is_group = list_[0].isupper()
        is_comment = list_[0] == '#'
    else:
        is_group = False
        is_comment = False

    # Finishing
    return is_empty, is_group, is_comment


def type_conversions(flag, value):
    """ Type conversions
    """
    # Type conversion
    if flag in ['seed', 'agents', 'maxfun']:
        value = int(value)
    elif flag in ['file', 'optimizer', 'start']:
        value = str(value)
    elif flag in ['detailed']:
        assert (value.upper() in ['TRUE', 'FALSE'])
        value = (value.upper() == 'TRUE')
    elif flag in []:
        value = value.upper()
    else:
        value = float(value)

    # Finishing
    return value
