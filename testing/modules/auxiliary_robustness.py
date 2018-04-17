"""This module contains auxiliary functions for the robustness testing."""
import os

import numpy as np

from interalpy.tests.test_auxiliary import get_random_init
from interalpy import estimate

# This requires access to a private repository that ensures that the data remains confidential.
DATA_PATH = os.environ['INTERTEMPORAL_ALTRUISM'] + '/sandbox/peisenha/structural_attempt/data'


def run_robustness_test(seed):
    """This function runs a single robustness test."""
    np.random.seed(seed)

    constr = dict()
    constr['est_file'] = DATA_PATH + '/risk_data_estimation.pkl'
    constr['num_agents'] = np.random.randint(1, 244 + 1)
    constr['maxfun'] = np.random.randint(500, 10000)

    get_random_init(constr)

    estimate('test.interalpy.ini')
