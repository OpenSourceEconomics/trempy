#!/usr/bin/env python
"""This module contains the capabilities to simulate the model."""
import shutil
import os

import pandas as pd
import numpy as np

from trempy.shared.shared_auxiliary import determine_optimal_compensation
from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.shared.shared import criterion_function
from trempy.clsModel import ModelCls


def simulate(fname):
    """This function simulates the model based on the initialization file."""

    model_obj = ModelCls(fname)

    sim_agents, questions, sim_seed, sim_file, paras_obj, cutoffs = dist_class_attributes(model_obj,
        'sim_agents', 'questions', 'sim_seed', 'sim_file', 'paras_obj', 'cutoffs')

    alpha, beta, eta = paras_obj.get_values('econ', 'all')[:3]

    # First, I simply determine the optimal compensations.
    m_optimal = []
    for q in questions:
        m_optimal += [determine_optimal_compensation(alpha, beta, eta, q)]

    np.random.seed(sim_seed)


    stands = paras_obj.get_values('econ', questions)

    data = []
    for i in range(sim_agents):
        for k, q in enumerate(sorted(questions)):
            m_latent = m_optimal[k] + np.random.normal(loc=0.0, scale=stands[k], size=1)[0]
            m_observed = m_latent

            # We need to account for the cutoffs.
            lower_cutoff, upper_cutoff = cutoffs[q]
            if m_latent < lower_cutoff:
                m_observed = lower_cutoff
            elif m_latent > upper_cutoff:
                m_observed = 9999

            data += [[i, q, m_observed]]

    df = pd.DataFrame(data)
    df.rename({0: 'Individual', 1: 'Question', 2: 'Compensation'}, inplace=True, axis='columns')
    dtype = {'Individual': np.int, 'Question': np.int, 'Compensation': np.float}
    df = df.astype(dtype)
    df.set_index(['Individual', 'Question'], inplace=True, drop=False)
    df.sort_index(inplace=True)

    df.to_pickle(sim_file + '.trempy.pkl', protocol=2)

    likl = criterion_function(df, questions, cutoffs, [alpha, beta, eta] + list(stands))

    write_info(df, questions, likl, m_optimal, sim_file + '.trempy.info')

    return df


def simulate_estimation(sim_agents, name, questions, cutoffs, paras):
    """This function allows for easy simulation of estimation results."""
    alpha, beta, eta = paras[:3]
    stands = paras[3:]

    if os.path.exists(name):
        shutil.rmtree(name)
    os.mkdir(name)
    os.chdir(name)

    df = simulate(sim_agents, questions, alpha, beta, eta, stands, cutoffs, fname='finish')
    os.chdir('../')

    return df


def write_info(df, questions, likl, m_optimal, fname):
    """This function writes out some basic information about the simulated dataset."""
    df_obs = df[abs(df['Compensation']) < 1000]

    # I need to introduce some logging.
    with open(fname, 'w') as outfile:
        string = '{:>15}' * 10 + '\n'
        label = []
        label += ['Question', 'Count', 'Mean', 'Optimal', 'Std.', 'Min.']
        label += ['25%', '50%', '75%', 'Max.']
        outfile.write(string.format(*label))
        outfile.write('\n')

        for i, q in enumerate(questions):
            stats = df_obs['Compensation'].loc[slice(None), slice(int(q), int(q))].describe(

            ).tolist()

            info = [q] + stats[:2] + [m_optimal[i]] + stats[2:]

            for i in [0, 1]:
                info[i] = '{:d}'.format(int(info[i]))
            for i in [2, 3, 4, 5, 6, 7, 8, 9]:
                info[i] = '{:15.5f}'.format(info[i])

            outfile.write(string.format(*info))

        outfile.write('\n')

        outfile.write('Criterion Function: ' + '{:15.5f}'.format(likl) + '\n\n')
