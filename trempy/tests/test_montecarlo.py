"""Contains the Monte Carlo simulation tests."""

import numpy as np
import copy

from trempy.shared.shared_auxiliary import print_init_dict
from trempy.config_trempy import PREFERENCE_PARAMETERS
from trempy.tests.test_auxiliary import random_dict
from trempy.custom_exceptions import TrempyError
from trempy.config_trempy import DEFAULT_BOUNDS
from trempy.read.read import ESTIMATION_GROUP
from trempy.estimate.estimate import estimate
from trempy.simulate.simulate import simulate
from trempy.config_trempy import SMALL_FLOAT
from trempy.read.read import read


def perturbate_all(init_dict, no_temporal_choices=True):
    """Perturbate all economic parameters and set bounds to default bounds."""
    old_dict = copy.deepcopy(init_dict)

    version = init_dict['VERSION']['version']

    for group in ESTIMATION_GROUP[version]:
        for label in PREFERENCE_PARAMETERS[version]:
            if label in init_dict[group].keys():
                # Distribute parameters
                value, is_fixed, bounds = init_dict[group][label]

                # Handle optional or unused arguments.
                if value is None:
                    continue
                lower, upper = DEFAULT_BOUNDS[label]

                # Get new value
                new_value = np.random.uniform(lower, upper)

                if group in ['DISCOUNTING'] and no_temporal_choices is True:
                    is_fixed = True
                    new_value = value
                else:
                    is_fixed = False

                # Update
                old_dict[group][label] = [value, is_fixed, [lower, upper]]
                init_dict[group][label] = [new_value, is_fixed, [lower, upper]]

    # Fix variances.
    for q in init_dict['QUESTIONS'].keys():
        init_dict['QUESTIONS'][q][1] = True
        old_dict['QUESTIONS'][q][1] = True

    return old_dict, init_dict


def pertubation_robustness_all(version, num_agents=None, maxfun=None, no_temporal_choices=True,
                               all_questions=True):
    """Test pertubation of all parameters."""
    # Get random init file
    constr = {'version': version, 'fname': 'perturb.all.truth', 'all_questions': all_questions}
    if num_agents is not None:
        constr['num_agents'] = num_agents
    if maxfun is None:
        constr['maxfun'] = 500
    else:
        constr['maxfun'] = maxfun

    init_dict = random_dict(constr)
    init_dict['ESTIMATION']['optimizer'] = 'SCIPY-POWELL'
    init_dict['SCIPY-POWELL']['ftol'] = 0.1
    init_dict['SCIPY-POWELL']['xtol'] = 0.01

    # Perturb parameters
    start_dict, perturbated_dict = perturbate_all(init_dict, no_temporal_choices)

    perturbated_dict

    # Save dicts
    print_init_dict(start_dict, 'perturb.all.truth')
    print_init_dict(perturbated_dict, 'perturb.all.perturbed')

    # Simulate data from init file
    simulate('perturb.all.truth')

    # Estimate starting from perturbed values
    estimate('perturb.all.perturbed')

    # os.chdir('stop')
    estimated_dict = read('stop/stop.trempy.ini')
    # os.chdir('../')

    for group in ESTIMATION_GROUP[version]:
        for key in init_dict[group].keys():
            start_value, is_fixed, bounds = start_dict[group][key]
            perturbed_value = perturbated_dict[group][key][0]
            estimated_value = estimated_dict[group][key][0]

            if start_value is None or is_fixed is True:
                continue

            print('{0:<25} {1:<15}'.format('Parameter:', key))
            print('-------------------------')
            print('{0:<25} {1:5.4f}'.format('Start:', start_value))
            print('{0:<25} {1:5.4f}'.format('Perturbated value:', perturbed_value))
            print('{0:<25} {1:5.4f}'.format('Estimated value:', estimated_value))


def perturbate_single(init_dict, label, value=None):
    """Perturbate a single parameter and fix all other parameters for estimation.

    We also set the bounds for the perturbed parameter to its default bounds.
    This increases the scope for perturbations.
    """
    old_dict = copy.deepcopy(init_dict)

    version = init_dict['VERSION']['version']
    if label not in PREFERENCE_PARAMETERS[version]:
        raise TrempyError('Version {0} has no parameters {1}'.format(version, label))

    # Fix variance for each question.
    for q in init_dict['QUESTIONS'].keys():
        init_dict['QUESTIONS'][q][1] = True

    # Handle optional parameters
    if label.startswith('unrestricted_weights'):
        not_used = (None in init_dict['TEMPORAL'].values())
        if not_used:
            raise TrempyError('Cannot set value for unused argument: {}.'.format(label))

    # Fix every parameter except for perturbed one. The perturbed one is "un-fixed".
    for group in ESTIMATION_GROUP[version]:
        for key in init_dict[group].keys():
            current_value, is_fixed, bounds = init_dict[group][key]
            if key == label:
                # Reset bounds to default
                lower, upper = DEFAULT_BOUNDS[label]
                # If no value is specified, draw a random value.
                if value is None:
                    value = np.random.uniform(lower + SMALL_FLOAT, upper - SMALL_FLOAT)
                init_dict[group][key] = [value, False, [lower, upper]]
                # Also, override old bounds in old dict.
                old_dict[group][key] = [current_value, False, [lower, upper]]
            # Fix all other parameters.
            else:
                init_dict[group][key] = [current_value, True, bounds]

    return old_dict, init_dict


def pertubation_robustness_single(version, label=None, value=None, num_agents=None, maxfun=None):
    """Check robustness against single perturbations."""
    if label is None:
        label = np.random.choice(PREFERENCE_PARAMETERS[version])

    # Get random init file
    constr = {'version': version, 'fname': 'perturb.start'}
    if num_agents is not None:
        constr['num_agents'] = num_agents
    if maxfun is None:
        constr['maxfun'] = 50
    else:
        constr['maxfun'] = maxfun
    init_dict = random_dict(constr)

    init_dict['ESTIMATION']['optimizer'] = 'SCIPY-POWELL'
    init_dict['ftol'] = 0.01
    init_dict['xtol'] = 0.05

    # Perturb parameters
    old_dict, perturbated = perturbate_single(init_dict, label=label, value=value)

    # Save dicts
    print_init_dict(old_dict, 'perturb.start')
    print_init_dict(perturbated, 'perturb.end')

    # Simulate data from init file
    simulate('perturb.start')

    # Estimate starting from perturbed values
    estimate('perturb.end')

    # os.chdir('stop')
    estimated_dict = read('stop/stop.trempy.ini')
    # os.chdir('../')

    for group in ESTIMATION_GROUP[version]:
        for key in init_dict[group].keys():
            if key == label:
                start_value = old_dict[group][key][0]
                perturbed_value = perturbated[group][key][0]
                estimated_value = estimated_dict[group][key][0]

    print('{0:<25} {1:<15}'.format('Parameter:', label))
    print('-------------------------')
    print('{0:<25} {1:5.4f}'.format('Start:', start_value))
    print('{0:<25} {1:5.4f}'.format('Perturbated value:', perturbed_value))
    print('{0:<25} {1:5.4f}'.format('Estimated value:', estimated_value))
