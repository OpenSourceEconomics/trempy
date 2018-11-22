"""This module contains auxiliary functions for the test runs."""
import shlex

import numpy as np

from trempy.shared.shared_auxiliary import get_random_string
from trempy.shared.shared_auxiliary import print_init_dict
from trempy.config_trempy import PREFERENCE_PARAMETERS
from trempy.custom_exceptions import TrempyError
from trempy.config_trempy import DEFAULT_BOUNDS
from trempy.config_trempy import HUGE_FLOAT


def get_random_init(constr=None):
    """Print a random dictionary."""
    if constr is None:
        constr = dict()

    init_dict = random_dict(constr)

    if 'fname' in constr.keys():
        fname = constr['fname']
        print_init_dict(init_dict, fname)
    else:
        print_init_dict(init_dict)

    return init_dict


def random_dict(constr):
    """Create a random initialization file."""
    dict_ = dict()

    # Handle version specific data.
    version = np.random.choice(['scaled_archimedean', 'nonstationary'])
    if constr is not None and 'version' in constr.keys():
        version = constr['version']

    dict_['VERSION'] = {'version': version}

    if 'all_questions' in constr.keys():
        num_questions = 16
    else:
        num_questions = np.random.randint(8, 14)
    sim_agents = np.random.randint(2, 10)

    if constr is not None and 'fname' in constr.keys():
        fname = constr['fname']
    else:
        fname = get_random_string()

    is_fixed = np.random.choice(
        [True, False], size=num_questions + len(PREFERENCE_PARAMETERS[version]))
    # We need to ensure at least one parameter is free for a valid estimation request.
    if is_fixed.tolist().count('False') == 0:
        is_fixed[0] = 'False'

    # Bounds and values. Be careful: the order of labels matters!
    bounds = [get_bounds(label, version) for label in PREFERENCE_PARAMETERS[version]]
    values = [get_value(bounds[i], label, version)
              for i, label in enumerate(PREFERENCE_PARAMETERS[version])]

    if version in ['scaled_archimedean']:
        # Initial setup to ensure constraints across options.
        marginals = np.random.choice(['exponential', 'power'], 2)
        upper_bounds = np.random.random_integers(500, 800, 2)

        # We start with sampling all preference parameters.
        dict_['UNIATTRIBUTE SELF'], i = dict(), 0
        dict_['UNIATTRIBUTE SELF']['r'] = [values[i], is_fixed[i], bounds[i]]
        dict_['UNIATTRIBUTE SELF']['max'] = upper_bounds[i]
        dict_['UNIATTRIBUTE SELF']['marginal'] = marginals[i]

        dict_['UNIATTRIBUTE OTHER'], i = dict(), 1
        dict_['UNIATTRIBUTE OTHER']['r'] = [values[i], is_fixed[i], bounds[i]]
        dict_['UNIATTRIBUTE OTHER']['max'] = upper_bounds[i]
        dict_['UNIATTRIBUTE OTHER']['marginal'] = marginals[i]

        dict_['MULTIATTRIBUTE COPULA'] = dict()
        for i, label in enumerate(['delta', 'self', 'other']):
            # We increment index because (r_self, r_other) are handled above.
            j = i + 2
            dict_['MULTIATTRIBUTE COPULA'][label] = [values[j], is_fixed[j], bounds[j]]

    elif version in ['nonstationary']:
        dict_['ATEMPORAL'] = dict()
        dict_['DISCOUNTING'] = dict()
        for i, label in enumerate(PREFERENCE_PARAMETERS[version]):
            if label in ['alpha', 'beta', 'gamma', 'y_scale']:
                dict_['ATEMPORAL'][label] = [values[i], is_fixed[i], bounds[i]]
            else:
                dict_['DISCOUNTING'][label] = [values[i], is_fixed[i], bounds[i]]

        # Handle optional arguments. If one argument is not used, set all to None and fix them.
        optional_args = ['unrestricted_weights_{}'.format(int(x)) for x in [0, 1, 3, 6, 12, 24]]
        not_used = (None in [dict_['DISCOUNTING'][label][0] for label in optional_args])
        if not_used:
            for label in optional_args:
                dict_['DISCOUNTING'][label] = [None, True, [0.01, 1.00]]

    else:
        raise TrempyError('version not implemented')

    # General part of the init file that does not change with the version.

    # It is time to sample the questions.
    questions = np.random.choice([13] + list(range(31, 46)), size=num_questions, replace=False)
    print(questions)
    print(num_questions)
    dict_['QUESTIONS'] = dict()

    for i, q in enumerate(questions):
        bounds = get_bounds(q, version)
        value = get_value(bounds, q, version)
        dict_['QUESTIONS'][q] = [value, is_fixed[i + len(PREFERENCE_PARAMETERS[version])], bounds]

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
    est_agents = np.random.randint(1, sim_agents)
    num_skip = np.random.randint(0, sim_agents - est_agents)

    dict_['ESTIMATION'] = dict()
    dict_['ESTIMATION']['optimizer'] = np.random.choice(['SCIPY-BFGS', 'SCIPY-POWELL'])
    dict_['ESTIMATION']['detailed'] = np.random.choice([True, False], p=[0.9, 0.1])
    dict_['ESTIMATION']['start'] = np.random.choice(['init', 'auto'])
    dict_['ESTIMATION']['agents'] = est_agents
    dict_['ESTIMATION']['skip'] = num_skip
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
            dict_['ESTIMATION']['skip'] = 0

        if 'est_file' in constr.keys():
            dict_['ESTIMATION']['file'] = constr['est_file']

        if 'detailed' in constr.keys():
            dict_['ESTIMATION']['detailed'] = constr['detailed']

        if 'start' in constr.keys():
            dict_['ESTIMATION']['start'] = constr['start']

    return dict_


def get_rmse():
    """Return the RMSE from the information file."""
    with open('compare.trempy.info') as in_file:
        for line in in_file.readlines():
            if 'RMSE' in line:
                stat = shlex.split(line)[1]
                if stat not in ['---']:
                    stat = float(stat)
                return stat


def get_bounds(label, version):
    """Return a set of valid bounds tailored for each parameter."""
    wedge = float(np.random.uniform(0.03, 0.10))

    # Questions
    if label in [13] + list(range(31, 46)):
        lower = float(np.random.uniform(0.01, 0.98 - wedge))
    else:
        # Handle version
        if version in ['scaled_archimedean']:
            if label in ['r_self', 'r_other']:
                lower = float(np.random.uniform(0.01, 5.0 - wedge))
            elif label in ['delta', 'self', 'other']:
                lower = float(np.random.uniform(0.01, 0.98 - wedge))
            else:
                raise TrempyError('flawed request for bounds')
        elif version in ['nonstationary']:
            if label in ['alpha', 'beta', 'gamma']:
                lower = float(np.random.uniform(0.01, 5.0 - wedge))
            elif label in ['y_scale']:
                lower = float(np.random.uniform(0.01, 0.98 - wedge))
            elif label.startswith('discount_factors'):
                lower = float(np.random.uniform(0.01, 0.98 - wedge))
            elif label.startswith('unrestricted_weights'):
                lower = float(np.random.uniform(0.01, 0.98 - wedge))
            else:
                raise TrempyError('flawed request for bounds')
        else:
            raise TrempyError('version not implemented')

    # Get upper bound by adding the wedge
    upper = lower + wedge

    # We want to check the case of the default bounds as well.
    if np.random.choice([True, False], p=[0.1, 0.9]):
        lower = DEFAULT_BOUNDS[label][0]
    if np.random.choice([True, False], p=[0.1, 0.9]):
        upper = DEFAULT_BOUNDS[label][1]

    bounds = [float(lower), float(upper)]

    bounds = [np.around(bound, decimals=4) for bound in bounds]

    return bounds


def get_value(bounds, label, version):
    """Return a value for the parameter that honors the bounds."""
    lower, upper = bounds

    if label in PREFERENCE_PARAMETERS[version]:
        # Handle optional arguments and set them to None if not required.
        if label.startswith('unrestricted_weights'):
            restricted = np.random.choice([True, False], p=[0.8, 0.2])
            if restricted:
                value = None
            else:
                value = float(np.random.uniform(lower + 0.01, upper - 0.01))
                value = np.around(value, decimals=4)
        # Other preference paramters
        else:
            value = float(np.random.uniform(lower + 0.01, upper - 0.01))
            value = np.around(value, decimals=4)
    # Handle non-preference labels
    else:
        upper = min(upper, 10)
        value = float(np.random.uniform(lower + 0.01, upper - 0.01))
        value = np.around(value, decimals=4)

    return value


def get_cutoffs():
    """Return a valid cutoff value."""
    lower = np.random.uniform(-5.0, -0.01)
    upper = np.random.uniform(0.01, 5.0)

    cutoffs = [np.random.choice([lower, -HUGE_FLOAT], p=[0.8, 0.2]),
               np.random.choice([upper, HUGE_FLOAT], p=[0.8, 0.2])]
    cutoffs = [np.around(cutoff, decimals=4) for cutoff in cutoffs]
    return cutoffs
