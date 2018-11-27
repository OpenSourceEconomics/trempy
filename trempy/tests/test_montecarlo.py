"""Contains the Monte Carlo simulation tests."""

import numpy as np
import copy

from trempy.estimate.estimate_auxiliary import estimate_cleanup
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


def basic_dict(version, fname, optimizer, maxfun, num_agents, std=None,
               eps=None, ftol=None, gtol=None):
    """Generate basic dictionary for Monte Carlo Simulations."""
    constr = {
        'version': version, 'fname': fname, 'num_agents': num_agents,
        'maxfun': maxfun, 'optimizer': optimizer, 'all_questions': True,
    }
    init_dict = random_dict(constr)

    # Add user-specified std deviations
    if std is not None:
        for q, sd in std.items():
            init_dict['QUESTIONS'][q][0] = sd

    # Handle optimizer options
    if eps is None:
        eps = 1e-05
    if ftol is None:
        ftol = 1e-08
    if gtol is None:
        gtol = 1e-08
    nuisance_paras = {'eps': eps, 'ftol': ftol, 'gtol': gtol}
    for label in ['eps', 'ftol', 'gtol']:
        if label in init_dict[optimizer].keys():
            init_dict[optimizer][label] = nuisance_paras[label]

    return init_dict


def set_questions(init_dict, is_fixed, std=None):
    """Manipulate questions."""
    # Change free and fixed status
    if is_fixed in ['fix_all']:
        for q in init_dict['QUESTIONS'].keys():
            init_dict['QUESTIONS'][q][1] = True
    else:
        np.testing.assert_equal(len(is_fixed), len(init_dict['QUESTIONS'].keys()))
        for q, fix_value in enumerate(is_fixed):
            init_dict['QUESTIONS'][q][1] = fix_value
    # Change standard deviations
    if std is not None:
        np.testing.assert_equal(len(std), len(init_dict['QUESTIONS'].keys()))
        for q, sd in enumerate(std):
            init_dict['QUESTIONS'][q][0] = sd


def remove_cutoffs(init_dict):
    """Remove cutoffs."""
    init_dict['CUTOFFS'] = dict()
    return dict


def estimate_at_truth(fix_question_paras):
    """Stability of the likelihood at the truth."""
    estimate_cleanup()

    init_dict = basic_dict(version='nonstationary', optimizer='SCIPY-L-BFGS-B', fname='truth',
                           num_agents=2000, maxfun=1000)

    set_questions(init_dict, is_fixed=fix_question_paras, std=None)

    seed = init_dict['SIMULATION']['seed']
    version = init_dict['VERSION']['version']

    print_init_dict(init_dict, fname='truth.trempy.ini')

    df, fval = simulate('truth.trempy.ini')
    est_output = estimate('truth.trempy.ini')

    # Print output
    estimated_dict = read('stop/stop.trempy.ini')

    results = list()

    for group in ESTIMATION_GROUP[version]:
        for key in init_dict[group].keys():
            start_value, is_fixed, bounds = init_dict[group][key]
            estimated_value = estimated_dict[group][key][0]

            if start_value is None or is_fixed is True:
                continue

            results.append([seed, fval, est_output[0], key, start_value, estimated_value])

            print('{0:<25} {1:<15}'.format('Parameter:', key))
            print('-------------------------')
            print('{0:<25} {1:5.4f}'.format('Truth:', start_value))
            print('{0:<25} {1:5.4f}'.format('Estimated value:', estimated_value))

    print(' ------------------------- ')
    print('sim seed: {:>25}'.format(seed))
    print('fval at truth: {:>25}'.format(fval))
    print(' ------------------------- ')

    return results


def perturbate_econ(init_dict, no_temporal_choices=True, max_dist=None):
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

                # Move the parameter by less than max_dist away.
                if max_dist is not None:
                    new_value = np.random.uniform(value - max_dist, value + max_dist)
                    new_value = min(upper, new_value)
                    new_value = max(lower, new_value)
                else:
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
    return old_dict, init_dict


def pertubation_robustness_all(version, no_temporal_choices=True,
                               max_dist=None, set_std_to=None):
    """Test pertubation of all parameters."""
    # Get random init file
    estimate_cleanup()

    init_dict = basic_dict(version=version, optimizer='SCIPY-L-BFGS-B', fname='truth',
                           num_agents=2000, maxfun=1000)

    # Set variance for questions
    if set_std_to is not None:
        for q in init_dict['QUESTIONS'].keys():
            init_dict['QUESTIONS'][q][0] = set_std_to
            init_dict['QUESTIONS'][q][2] = [set_std_to - SMALL_FLOAT, set_std_to + SMALL_FLOAT]

    set_questions(init_dict, is_fixed='fix_all', std=None)

    seed = init_dict['SIMULATION']['seed']
    version = init_dict['VERSION']['version']

    print_init_dict(init_dict, fname='truth.trempy.ini')

    # Perturb parameters
    truth_dict, perturbed_dict = perturbate_econ(
        init_dict, no_temporal_choices=no_temporal_choices, max_dist=max_dist)

    print_init_dict(perturbed_dict, fname='perturbed.trempy.ini')

    # Simulate data from init file and report criterion function.
    df, fval = simulate('truth.trempy.ini')
    print('fval at truth: {:>25}'.format(fval))

    # Estimate starting from perturbed values
    estimate('perturbed.trempy.ini')
    estimated_dict = read('stop/stop.trempy.ini')

    for group in ESTIMATION_GROUP[version]:
        for key in init_dict[group].keys():
            start_value, is_fixed, bounds = truth_dict[group][key]
            perturbed_value = perturbed_dict[group][key][0]
            estimated_value = estimated_dict[group][key][0]

            if start_value is None or is_fixed is True:
                continue

            print('{0:<25} {1:<15}'.format('Parameter:', key))
            print('-------------------------')
            print('{0:<25} {1:5.4f}'.format('Start:', start_value))
            print('{0:<25} {1:5.4f}'.format('Perturbated value:', perturbed_value))
            print('{0:<25} {1:5.4f}'.format('Estimated value:', estimated_value))

    print('Seed: {:>25}'.format(seed))
    print('fval_truth: {:>25}'.format(fval))


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


def pertubation_robustness_single(version, label=None, value=None, num_agents=None, maxfun=None,
                                  optimizer='SCIPY-BFGS'):
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

    init_dict['ESTIMATION']['optimizer'] = optimizer
    init_dict['SCIPY-POWELL']['ftol'] = 0.1
    init_dict['SCIPY-POWELL']['xtol'] = 0.01

    init_dict['SCIPY-BFGS']['eps'] = 1.4901161193847656e-08
    init_dict['SCIPY-BFGS']['gtol'] = 1e-05

    init_dict['SCIPY-L-BFGS-B']['eps'] = 1.4901161193847656e-08
    init_dict['SCIPY-L-BFGS-B']['gtol'] = 1.5e-08
    init_dict['SCIPY-L-BFGS-B']['ftol'] = 1.5e-08

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
