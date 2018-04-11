"""This module contains auxiliary functions for the test runs."""
import linecache
import shlex

import numpy as np

from trempy.shared.shared_auxiliary import get_random_string
from trempy.shared.shared_auxiliary import print_init_dict
from trempy.custom_exceptions import TrempyError
#from interalpy.config_interalpy import NUM_PARAS
from trempy.config_trempy import HUGE_FLOAT


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
    is_fixed = np.random.choice(['True', 'False'], size=num_questions + 3)

    # We need to ensure at least one parameter is free for a valid estimation request.
    if is_fixed.tolist().count('False') == 0:
        is_fixed[0] = 'False'

    bounds = list()
    for label in ['alpha', 'beta', 'eta']:
        bounds += [get_bounds(label)]

    values = list()
    for i, label in enumerate(['alpha', 'beta', 'eta']):
        values += [get_value(bounds[i])]

    # We want to include the case where the bounds are not specified by the user. In this case
    # the default bounds are relevant.
    probs = [0.2, 0.8]
    for bound in bounds:
        if np.random.choice([True, False], p=probs):
            bound[0] = -np.inf
        if np.random.choice([True, False], p=probs):
            bound[1] = np.inf

    # We start with sampling all preference parameters.
    dict_['PREFERENCES'] = dict()
    for i, label in enumerate(['alpha', 'beta', 'eta']):
         dict_['PREFERENCES'][label] = (values[i], is_fixed[i], bounds[i])

    # It is time to sample the questions.
    dict_['QUESTIONS'] = dict()

    for i, q in enumerate(questions):
        bounds = get_bounds(None)
        value = get_value(bounds)
        dict_['QUESTIONS'][q] = (value, is_fixed[i + 3], bounds)

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
    dict_['ESTIMATION']['optimizer'] = np.random.choice(['SCIPY-BFGS'])
    dict_['ESTIMATION']['detailed'] = 'True'
    dict_['ESTIMATION']['start'] = np.random.choice(['auto', 'init'])
    dict_['ESTIMATION']['agents'] = np.random.randint(1, sim_agents)
    dict_['ESTIMATION']['maxfun'] = np.random.randint(1, 10)
    dict_['ESTIMATION']['file'] = fname + '.trempy.pkl'

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
    # These are the cutoff values.
    elif label is None:
        lower = float(np.random.uniform(0.01, 0.98 - wedge))
        upper = lower + wedge
    else:
        raise TrempyError('flawed request for bounds')

    bounds = [float(lower), float(upper)]

    return bounds


def get_value(bounds):
    """This function returns a value for the parameter that honors the bounds."""
    lower, upper = bounds
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
