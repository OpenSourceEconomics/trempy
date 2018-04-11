"""This module contains function solely related to the estimation of the model."""
import shutil
import copy
import os

from statsmodels.tools.eval_measures import rmse as get_rmse
import pandas as pd
import numpy as np

from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.config_trempy import HUGE_FLOAT
from trempy.simulate.simulate import simulate


def estimate_cleanup():
    """This function ensures that we start the estimation with a clean slate."""
    # We remove the directories that contain the simulated choice menus at the start.
    for dirname in ['start', 'stop']:
        if os.path.exists(dirname):
            shutil.rmtree(dirname)

    # We remove the information from earlier estimation runs.
    for fname in ['est.trempy.info', 'est.trempy.log']:
        if os.path.exists(fname):
            os.remove(fname)


def estimate_simulate(which, points, model_obj, df_obs):
    """This function allows to easily simulate samples at the beginning and the end of the
    estimation."""
    questions = dist_class_attributes(model_obj, 'questions')

    os.mkdir(which)
    os.chdir(which)

    sim_model = copy.deepcopy(model_obj)
    sim_model.attr['sim_file'] = which

    sim_model.update('optim', 'free', points)
    sim_model.write_out(which + '.trempy.ini')
    simulate(which + '.trempy.ini')

    compare_datasets(which, df_obs, questions)

    os.chdir('../')


def compare_datasets(which, df_obs, questions):
    """This function compares the estimation dataset with a simulated dataset using the estimated
    parameter vector."""
    df_sim = pd.read_pickle(which + '.trempy.pkl')

    df_sim_masked = df_sim['Compensation'].mask(df_sim['Compensation'] == 9999)
    df_obs_masked = df_obs['Compensation'].mask(df_obs['Compensation'] == 9999)

    stats = dict()
    stats['sim'] = []
    for q in questions:
        num_obs = df_sim.loc[(slice(None), slice(q, q)), 'Compensation'].shape[0]
        stats['sim'] += [[num_obs] + df_sim_masked.loc[slice(None), slice(q, q)].describe().tolist()]

    stats['obs'] = []
    for q in questions:
        num_obs = df_obs.loc[(slice(None), slice(q, q)), 'Compensation'].shape[0]
        stats['obs'] += [[num_obs] + df_obs_masked.loc[slice(None), slice(q, q)].describe().tolist()]

    with open('compare.trempy.info', 'w') as outfile:

        outfile.write('\n')
        string = '{:>15}' * 11 + '\n'

        label = []
        label += ['', 'Question', 'Observed', 'Interior', 'Mean', 'Std.', 'Min.']
        label += ['25%', '50%', '75%', 'Max.']

        outfile.write(string.format(*label))
        outfile.write('\n')

        for i, q in enumerate(questions):

            for key_ in ['obs', 'sim']:

                if key_ == 'obs':
                    label = 'Observed'
                elif key_ == 'sim':
                    label = 'Simulated'

                info = [label, q] + stats[key_][i]

                for j in range(len(info)):
                    if pd.isnull(info[j]):
                        info[j] = '{:>15}'.format('---')
                        continue

                    if j in [1, 2, 3]:
                        info[j] = '{:d}'.format(int(info[j]))

                    if j in [4, 5, 6, 7, 8, 9, 10]:
                        info[j] = '{:15.5f}'.format(info[j])

                outfile.write(string.format(*info))

            outfile.write('\n')

        # TODO: Transform to numpy array and first and then used masked version.
        mean_obs, mean_sim = [], []
        for i, q in enumerate(questions):

            is_nan = []
            is_nan += [np.isnan(stats['obs'][i][2])]
            is_nan += [np.isnan(stats['sim'][i][2])]

            if np.any(is_nan):
                continue
            mean_obs += [stats['obs'][i][2]]
            mean_sim += [stats['sim'][i][2]]

        rmse = get_rmse(mean_obs, mean_sim)

        line = '{:>15}'.format('RMSE') + '{:15.5f}\n'.format(rmse)
        outfile.write(line)


def char_floats(floats):
    """This method ensures a pretty printing of all floats."""
    # We ensure that this function can also be called on for a single float value.
    if isinstance(floats, float):
        floats = [floats]

    line = []
    for value in floats:
        if abs(value) > HUGE_FLOAT:
            line += ['{:>25}'.format('---')]
        else:
            line += ['{:25.15f}'.format(value)]

    return line
