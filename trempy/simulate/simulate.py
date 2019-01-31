#!/usr/bin/env python
"""This module contains the capabilities to simulate the model."""
import pandas as pd
import numpy as np

from trempy.shared.shared_auxiliary import get_optimal_compensations
from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.shared.shared_auxiliary import criterion_function
from trempy.config_trempy import PREFERENCE_PARAMETERS
from trempy.config_trempy import NEVER_SWITCHERS
from trempy.custom_exceptions import TrempyError
from trempy.clsModel import ModelCls


def simulate(fname):
    """Simulate the model based on the initialization file."""
    model_obj = ModelCls(fname)
    version = model_obj.attr['version']

    # Get fixed args that do not change during simulation.
    args = [model_obj, 'sim_agents', 'questions', 'sim_seed', 'sim_file', 'paras_obj', 'cutoffs']
    if version in ['scaled_archimedean']:
        args += ['upper', 'marginals']
        sim_agents, questions, sim_seed, sim_file, paras_obj, cutoffs, upper, marginals = \
            dist_class_attributes(*args)

        version_specific = {'upper': upper, 'marginals': marginals}
    elif version in ['nonstationary']:
        sim_agents, questions, sim_seed, sim_file, paras_obj, cutoffs = \
            dist_class_attributes(*args)
        version_specific = dict()
    else:
        raise TrempyError('version not implemented')

    np.random.seed(sim_seed)
    m_optimal = get_optimal_compensations(version=version, paras_obj=paras_obj,
                                          questions=questions, **version_specific)

    # First, get number of preference parameters. Paras with higher index belong to questions!
    nparas_econ = paras_obj.attr['nparas_econ']
    # Now, get standard deviation for the error in each question. This handles versions implicitly.
    stands = paras_obj.get_values('econ', 'all')[nparas_econ:]

    # Simulate data
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

    # Post-processing step
    df = pd.DataFrame(data)
    df.rename({0: 'Individual', 1: 'Question', 2: 'Compensation'}, inplace=True, axis='columns')
    dtype = {'Individual': np.int, 'Question': np.int, 'Compensation': np.float}
    df = df.astype(dtype)
    df.set_index(['Individual', 'Question'], inplace=True, drop=False)
    df.sort_index(inplace=True)

    df.to_pickle(sim_file + '.trempy.pkl', protocol=2)

    x_econ_all_current = paras_obj.get_values('econ', 'all')

    fval = criterion_function(df=df, questions=questions, cutoffs=cutoffs,
                              paras_obj=paras_obj, version=version, sds=stands,
                              **version_specific)

    write_info(version, x_econ_all_current, df, questions,
               fval, m_optimal, sim_file + '.trempy.info')

    return df, fval


def write_info(version, x_econ_all_current, df, questions, likl, m_optimal, fname):
    """Write out some basic information about the simulated dataset."""
    df_sim = df['Compensation'].mask(df['Compensation'] == NEVER_SWITCHERS)
    paras_label = PREFERENCE_PARAMETERS[version] + questions
    fmt_ = '{:>15}' + '{:>15}' + '{:>15}    '

    with open(fname, 'w') as outfile:

        outfile.write('\n {:<25}\n'.format('Observed Data'))

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

        # Print economic parameters
        fmt_ = '{:>15}' + '{:>25}' + '{:>15}'
        outfile.write('\n {:<25}\n\n'.format('Economic Parameters'))
        line = ['Identifier', 'Label', 'Value']
        outfile.write(fmt_.format(*line) + '\n\n')
        for i, _ in enumerate(range(len(questions) + len(PREFERENCE_PARAMETERS[version]))):
            line = [i]

            # Handle optional arguments where None value marks optionality.
            if x_econ_all_current[i] is None:
                continue
            # Print all other parameters
            else:
                line += [paras_label[i], '{:15.5f}'.format(x_econ_all_current[i])]
                outfile.write(fmt_.format(*line) + '\n')

        outfile.write('\n')

        outfile.write(' Criterion Function ' + '{:25.5f}'.format(likl) + '\n\n')
