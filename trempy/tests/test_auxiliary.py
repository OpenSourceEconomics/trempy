"""This module contains auxiliary functions for the test runs."""
import shlex

import numpy as np

from trempy.shared.shared_auxiliary import get_random_string
from trempy.shared.shared_auxiliary import print_init_dict
from trempy.custom_exceptions import TrempyError
from trempy.config_trempy import DEFAULT_BOUNDS


def get_random_init(constr=None):
    """This function prints a random dictionary."""
    if constr is None:
        constr = dict()
    init_dict = random_dict(constr)
    print_init_dict(init_dict)
    return init_dict


def random_dict(constr):
    """This function creates a random initialization file."""
    dict_ = dict()

    # Initial setup to ensure constraints across options.
    num_questions = np.random.randint(2, 14)
    sim_agents = np.random.randint(2, 10)
    fname = get_random_string()

    questions = np.random.choice(range(1, 16), size=num_questions, replace=False)
    is_fixed = np.random.choice([True, False], size=num_questions + 3)

    # We need to ensure at least one parameter is free for a valid estimation request.
    if is_fixed.tolist().count('False') == 0:
        is_fixed[0] = 'False'

    bounds = list()
    for label in ['alpha', 'beta', 'eta']:
        bounds += [get_bounds(label)]

    values = list()
    for i, label in enumerate(['alpha', 'beta', 'eta']):
        values += [get_value(bounds[i], label)]

    # We start with sampling all preference parameters.
    dict_['PREFERENCES'] = dict()
    for i, label in enumerate(['alpha', 'beta', 'eta']):
         dict_['PREFERENCES'][label] = [values[i], is_fixed[i], bounds[i]]

    # It is time to sample the questions.
    dict_['QUESTIONS'] = dict()

    for i, q in enumerate(questions):
        bounds = get_bounds(q)
        value = get_value(bounds, q)
        dict_['QUESTIONS'][q] = [value, is_fixed[i + 3], bounds]

    # We now add some cutoff values.
    dict_['CUTOFFS'] = dict()
    for q in questions:
        if np.random.choice([True, False]):
            continue
        else:
            dict_['CUTOFFS'][q] = get_cutoffs()

    # We now turn to all simulation details.
    dict_['SIMULATION'] = dict()
    dict_['SIMULATION']['agents'] = sim_agents
    dict_['SIMULATION']['seed'] = np.random.randint(1, 1000)
    dict_['SIMULATION']['file'] = fname

    # We sample valid estimation requests.
    dict_['ESTIMATION'] = dict()
    dict_['ESTIMATION']['optimizer'] = np.random.choice(['SCIPY-BFGS', 'SCIPY-POWELL'])
    dict_['ESTIMATION']['detailed'] = True
    dict_['ESTIMATION']['start'] = np.random.choice(['auto', 'init'])
    dict_['ESTIMATION']['agents'] = np.random.randint(1, sim_agents)
    dict_['ESTIMATION']['maxfun'] = np.random.randint(1, 10)
    dict_['ESTIMATION']['file'] = fname + '.trempy.pkl'

    # We sample optimizer options.
    dict_['SCIPY-BFGS'] = dict()
    dict_['SCIPY-BFGS']['gtol'] = np.random.lognormal()
    dict_['SCIPY-BFGS']['eps'] = np.random.lognormal()

    dict_['SCIPY-POWELL'] = dict()
    dict_['SCIPY-POWELL']['xtol'] = np.random.lognormal()
    dict_['SCIPY-POWELL']['ftol'] = np.random.lognormal()

    # Now we need to impose possible constraints.
    if constr is not None:
        if 'maxfun' in constr.keys():
            dict_['ESTIMATION']['maxfun'] = constr['maxfun']

        if 'num_agents' in constr.keys():
            dict_['SIMULATION']['agents'] = constr['num_agents']
            dict_['ESTIMATION']['agents'] = constr['num_agents']

        if 'est_file' in constr.keys():
            dict_['ESTIMATION']['file'] = constr['est_file']

        if 'detailed' in constr.keys():
            dict_['ESTIMATION']['detailed'] = constr['detailed']

    return dict_


def get_rmse():
    """This function returns the RMSE from the information file."""
    with open('compare.trempy.info') as in_file:
        for line in in_file.readlines():
            if 'RMSE' in line:
                stat = float(shlex.split(line)[1])
                return stat


def get_bounds(label):
    """This function returns a set of valid bounds tailored for each parameter."""
    wedge = float(np.random.uniform(0.03, 0.10))

    if label in ['alpha', 'eta']:
        lower = float(np.random.uniform(0.01, 0.98 - wedge))
        upper = lower + wedge
    elif label in ['beta']:
        lower = float(np.random.uniform(0.00, 0.98 - wedge))
        upper = lower + wedge
    elif label in range(1, 16):
        lower = float(np.random.uniform(0.01, 0.98 - wedge))
        upper = lower + wedge
    else:
        raise TrempyError('flawed request for bounds')

    # We want to check the case of the default bounds as well.
    if np.random.choice([True, False], p=[0.1, 0.9]):
        lower = DEFAULT_BOUNDS[label][0]
    if np.random.choice([True, False], p=[0.1, 0.9]):
        upper = DEFAULT_BOUNDS[label][1]

    bounds = [float(lower), float(upper)]

    return bounds


def get_value(bounds, label):
    """This function returns a value for the parameter that honors the bounds."""
    lower, upper = bounds

    if label in ['alpha', 'beta', 'eta']:
        value = float(np.random.uniform(lower + 0.01, upper - 0.01))
    else:
        upper = min(upper, 10)
        value = float(np.random.uniform(lower + 0.01, upper - 0.01))

    return value


def get_cutoffs():
    """This function returns a valid cutoff value."""
    lower = np.random.uniform(-5.0, -0.01)
    upper = np.random.uniform(0.01, 5.0)

    cutoffs = []
    cutoffs += [np.random.choice([lower, -np.inf], p=[0.8, 0.2])]
    cutoffs += [np.random.choice([upper, np.inf], p=[0.8, 0.2])]

    return cutoffs
