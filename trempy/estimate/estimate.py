#!/usr/bin/env python
from functools import partial
import copy

import pandas as pd
import numpy as np

from trempy.estimate.estimate_auxiliary import write_info_estimation
from trempy.simulate.simulate import simulate_estimation
from trempy.shared.shared_auxiliary import criterion_function
from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.clsModel import ModelCls
from trempy.estimate.clsEstimate import EstimateClass


def estimate(fname):
    """This function estimates the model by the method of maximum likelihood."""

    model_obj = ModelCls(fname)

    est_file, questions, paras_obj, start, cutoffs, maxfun = dist_class_attributes(model_obj,
        'est_file', 'questions', 'paras_obj', 'start', 'cutoffs', 'maxfun')

    # Some initial setup
    df_obs = pd.read_pickle(est_file)

    # We drop all individuals that never switch between lotteries and restrict attention to a
    # subset of individuals.
    df_obs = df_obs[abs(df_obs['Compensation']) < 1000]
    cond = df_obs['Question'].isin(questions)
    df_obs = df_obs[cond]

    x_optim_free_start = paras_obj.get_values('optim', 'free')
    print(x_optim_free_start)

    estimate_obj = EstimateClass(df_obs, cutoffs, questions, copy.deepcopy(paras_obj), maxfun)

    estimate_obj.evaluate(x_optim_free_start)

    raise AssertionError

    # Construction of starting values
    if start == 'init':
        paras_start = paras_obj.get_values('econ', ['alpha', 'beta', 'eta'] + questions)
    elif start == 'auto':
        paras_start = []
        for label in ['alpha', 'beta', 'eta']:
            value, is_fixed = paras_obj.get_para(label)[:2]
            if is_fixed:
                continue

            paras_start += [value]

        for q in questions:
            q = int(q)
            paras_start += [np.sqrt(np.var(df_obs['Compensation'].loc[slice(None), slice(q, q)]))]

    else:
        raise NotImplementedError

    # Add parameter bounds
    paras_bounds = []
    for label in ['alpha', 'beta', 'eta'] + questions:
        paras_bounds += [paras_obj.get_para(label)[2]]

    # Taking stock of initial distribution
    # paras_si = paras_obj.get_values('econ', ['alpha', 'beta', 'eta'] + questions)
    # simulate_estimation('start', questions, cutoffs, paras_simulation)
    cutoffs = None

    # Optimization of likelihood function
    criterion_function = partial(likelihood, df_obs, questions, cutoffs)
    if False:
        rslt = fmin_l_bfgs_b(criterion_function, paras_start, bounds=bounds, approx_grad=True,
                maxfun=1)
    else:
        criterion_function(paras_start)
        #rslt = []
        #rslt += [paras_start]
        #rslt += [criterion_function(paras_start)]

    # # Detailed inspection of results
    # j = 0
    # paras_simulation = utility_paras_start
    # for i in range(3):
    #     if utility_paras_free[i]:
    #         paras_simulation[i] = rslt[0][j]
    #         j += 1
    #
    # paras_simulation += rslt[0][num_utility_paras_free:]
    #
    # df_finish = simulate_estimation('finish', questions, cutoffs, paras_simulation)
    # write_info_estimation(df_obs, df_finish)
    #
    print(rslt)
    # return rslt[1], rslt[0]
