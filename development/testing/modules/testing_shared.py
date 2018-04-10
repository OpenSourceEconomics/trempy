"""This module contains functions that are shared across the different tests."""
import shutil
import glob
import sys
import os
sys.path.insert(0, '../../../')

import numpy as np

from trempy import simulate, estimate


def cleanup(is_create=False):
    """This function cleans the directory from nuisance files."""
    keep = ['acropolis.pbs', 'run.py', 'regression_vault.interalpy.json']
    if is_create:
        keep += ['created_vault.interalpy.json']

    for fname in glob.glob("*"):
        if fname in keep:
            continue
        try:
            os.unlink(fname)
        except IsADirectoryError:
            shutil.rmtree(fname)


def create_random_init():
    """This function creates a random initialization dictionary."""
    num_questions = np.random.randint(1, 7)

    init_dict = dict()

    init_dict['num_agents'] = int(np.random.randint(10, 100))
    init_dict['alpha'] = float(np.random.uniform(0.01, 0.99))
    init_dict['seed'] = int(np.random.randint(10, 1000))
    init_dict['beta'] = float(np.random.uniform(0.01, 0.99))
    init_dict['eta'] = float(np.random.uniform(0.01, 0.99))

    init_dict['questions'] = np.random.choice([1, 2, 3, 10, 11, 12], replace=False,
        size=num_questions).tolist()

    init_dict['stands'] = np.random.uniform(5.0, 10.0, size=num_questions).tolist()

    init_dict['cutoffs'] = dict()
    for q in init_dict['questions']:
        lower = np.random.uniform(-15.0, 0.0)
        upper = lower + np.random.uniform(10.0, 20.0)
        init_dict['cutoffs'][q] = [lower, upper]

    return init_dict


def distribute_random_init(init_dict):
    """This function distributes an initialization dictionary."""
    rslt = []
    for key_ in ['num_agents', 'questions', 'alpha', 'beta', 'eta', 'stands', 'cutoffs', 'seed']:
        rslt += [init_dict[key_]]
    return rslt


def run_test_case(init_dict):
    """This function runs a test case."""
    num_agents, questions, alpha, beta, eta, stands, cutoffs, seed = \
        distribute_random_init(init_dict)
    simulate(num_agents, questions, alpha, beta, eta, stands, cutoffs, seed, 'data')
    fval, _ = estimate('data', cutoffs=cutoffs, subset=questions)

    return fval