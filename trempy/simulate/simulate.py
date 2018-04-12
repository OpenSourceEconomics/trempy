#!/usr/bin/env python
"""This module contains the capabilities to simulate the model."""
import pandas as pd
import numpy as np

from trempy.shared.shared_auxiliary import get_optimal_compensations
from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.shared.shared_auxiliary import criterion_function
from trempy.config_trempy import NEVER_SWITCHERS
from trempy.clsModel import ModelCls


def simulate(fname):
    """This function simulates the model based on the initialization file."""

    model_obj = ModelCls(fname)

    sim_agents, questions, sim_seed, sim_file, paras_obj, cutoffs = dist_class_attributes(model_obj,
        'sim_agents', 'questions', 'sim_seed', 'sim_file', 'paras_obj', 'cutoffs')

    np.random.seed(sim_seed)

    # First, I simply determine the optimal compensations.
    alpha, beta, eta = paras_obj.get_values('econ', 'all')[:3]

    m_optimal = get_optimal_compensations(questions, alpha, beta, eta)

    stands = paras_obj.get_values('econ', 'all')[3:]

    data = []
    for i in range(sim_agents):
        for k, q in enumerate(questions):
            m_latent = m_optimal[q] + np.random.normal(loc=0.0, scale=stands[k], size=1)[0]
            m_observed = m_latent

            # We need to account for the cutoffs.
            lower_cutoff, upper_cutoff = cutoffs[q]
            if m_latent < lower_cutoff:
                m_observed = lower_cutoff
            elif m_latent > upper_cutoff:
                m_observed = NEVER_SWITCHERS

            data += [[i, q, m_observed]]

    df = pd.DataFrame(data)
    df.rename({0: 'Individual', 1: 'Question', 2: 'Compensation'}, inplace=True, axis='columns')
    dtype = {'Individual': np.int, 'Question': np.int, 'Compensation': np.float}
    df = df.astype(dtype)
    df.set_index(['Individual', 'Question'], inplace=True, drop=False)
    df.sort_index(inplace=True)

    df.to_pickle(sim_file + '.trempy.pkl', protocol=2)

    x_econ_all_current = paras_obj.get_values('econ', 'all')

    fval = criterion_function(df, questions, cutoffs, *x_econ_all_current)

    write_info(df, questions, fval, m_optimal, sim_file + '.trempy.info')

    return df


def write_info(df, questions, likl, m_optimal, fname):
    """This function writes out some basic information about the simulated dataset."""
    df_sim = df['Compensation'].mask(df['Compensation'] == NEVER_SWITCHERS)

    with open(fname, 'w') as outfile:

        string = '{:>15}' * 10 + '\n'
        label = []
        label += ['Question', 'Observed', 'Interior', 'Optimal', 'Mean', 'Std.', 'Min.']
        label += ['25%', '50%', '75%', 'Max.']

        outfile.write('\n')
        outfile.write(string.format(*label))
        outfile.write('\n')

        for i, q in enumerate(questions):

            num_observed = df.loc[(slice(None), slice(q, q)), :].shape[0]
            stats = df_sim.loc[slice(None), slice(q, q)].describe().tolist()
            info = [q, num_observed, stats[0], m_optimal[q]] + stats[1:]

            for i in [0, 1, 2]:
                info[i] = '{:d}'.format(int(info[i]))
            for i in [3, 4, 5, 6, 7, 8, 9]:
                if pd.isnull(info[i]):
                    info[i] = '{:>15}'.format('---')
                else:
                    info[i] = '{:15.5f}'.format(info[i])

            outfile.write(string.format(*info))

        outfile.write('\n')

        outfile.write('Criterion Function: ' + '{:16.5f}'.format(likl) + '\n\n')
