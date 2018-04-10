#!/usr/bin/env python
import sys
sys.path.insert(0, '../../')

import numpy as np

from trempy import estimate
from trempy import simulate

from trempy.clsModel import ModelCls

from trempy.tests.test_auxiliary import random_dict
from trempy.shared.shared_auxiliary import print_init_dict
from trempy.tests.test_auxiliary import get_random_init
from trempy.read.read import read
from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy import simulate


for _ in range(1):
    get_random_init()
    simulate('test.trempy.ini')

# TODO: UNit tests for random simulation and printing.
#model_obj = ModelCls('model.trempy.ini')
#simulate('model.trempy.ini')
#estimate('model.trempy.ini')
# sys.exit('exit')
# if True:
#
#     np.random.seed(213)
#     # These are the current results from the empirical data.
#     num_agents = 100
#     questions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#     alpha = 0.10
#     beta = 0.50
#     eta = 0.10
#     stands = [0.1] * 15
#
#     cutoffs = None
#
#     utility_paras_free = [True, False, True]
#     utility_paras_start = [alpha, beta, eta]
#     df_obs = simulate(num_agents, questions, alpha, beta, eta, stands, cutoffs, fname='data')
#     fval, _ = estimate('data', utility_paras_start, utility_paras_free, cutoffs, questions)
#
#     np.testing.assert_equal(fval, 102.97587385750849)
#
#
# else:
#
# #    subset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#
#     subset = [1, 2, 3, 7, 8, 9]
#     cutoffs = dict()
#
#     cutoffs[1] = [-5.0, 5.0]
#     cutoffs[2] = [-10.0, 10.0]
#     cutoffs[3] = [-20.0, 20.0]
#
#     cutoffs[4] = [-5.0, 5.0]
#     cutoffs[5] = [-10.0, 10.0]
#     cutoffs[6] = [-20.0, 20.0]
#
#     cutoffs[7] = [-5.0, 5.0]
#     cutoffs[8] = [-10.0, 10.0]
#     cutoffs[9] = [-20.0, 20.0]
#
#     cutoffs[10] = [-5.0, 5.0]
#     cutoffs[11] = [-5.0, 5.0]
#     cutoffs[12] = [-5.0, 5.0]
#
#     cutoffs[13] = [-5.0, 5.0]
#     cutoffs[14] = [-5.0, 5.0]
#     cutoffs[15] = [-5.0, 5.0]
#
#     fval, _ = estimate('../../data/observed', cutoffs=cutoffs, subset=subset)
#
#     np.testing.assert_equal(fval, 2.5939466198008958)
