"""This module contains function solely related to the estimation of the model."""
import shutil
import copy
import os

from statsmodels.tools.eval_measures import rmse as get_rmse
import pandas as pd
import numpy as np

from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.config_trempy import NEVER_SWITCHERS
from trempy.simulate.simulate import simulate
from trempy.config_trempy import SMALL_FLOAT
from trempy.config_trempy import HUGE_FLOAT


def get_automatic_starting_values(paras_obj, df_obs, questions):
    """This method updates the container for the parameters with the automatic starting values."""
    def _adjust_bounds(value, bounds):
        """This function simply adjusts the starting values to meet the requirements of the
        bounds."""
        lower, upper = bounds
        if value <= bounds[0]:
            value = lower + 2 * SMALL_FLOAT
        elif value >= bounds[1]:
            value = upper - 2 * SMALL_FLOAT
        else:
            pass

        return value

    x_econ_free_start = []

    for label in ['alpha', 'beta', 'eta'] + questions:
        value, is_fixed, bounds = paras_obj.get_para(label)

        if is_fixed:
            continue
        else:
            if label in ['alpha', 'beta', 'eta']:
                x_econ_free_start += [_adjust_bounds(0.5, bounds)]
            else:
                df_mask = df_obs['Compensation'].mask(df_obs['Compensation'] == NEVER_SWITCHERS)
                value = df_mask.loc[slice(None), label].std()
                # If there are no individuals observed without truncation we start with 0.1.
                if pd.isnull(value):
                    x_econ_free_start += [_adjust_bounds(0.1, bounds)]
                else:
                    x_econ_free_start += [_adjust_bounds(value, bounds)]

    paras_obj.set_values('econ', 'free', x_econ_free_start)

    return paras_obj


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

    df_sim_masked = df_sim['Compensation'].mask(df_sim['Compensation'] == NEVER_SWITCHERS)
    df_obs_masked = df_obs['Compensation'].mask(df_obs['Compensation'] == NEVER_SWITCHERS)

    stats = dict()
    stats['sim'] = dict()
    for q in questions:
        num_obs = df_sim.loc[(slice(None), slice(q, q)), 'Compensation'].shape[0]
        stats['sim'][q] = [num_obs] + df_sim_masked.loc[slice(None), slice(q, q)].describe(

        ).tolist()

    stats['obs'] = dict()
    for q in questions:
        num_obs = df_obs.loc[(slice(None), slice(q, q)), 'Compensation'].shape[0]
        stats['obs'][q] = [num_obs] + df_obs_masked.loc[slice(None), slice(q, q)].describe(

        ).tolist()

    with open('compare.trempy.info', 'w') as outfile:

        outfile.write('\n')
        string = '{:>15}' * 11 + '\n'

        label = []
        label += ['', 'Question', 'Observed', 'Interior', 'Mean', 'Std.', 'Min.']
        label += ['25%', '50%', '75%', 'Max.']

        outfile.write(string.format(*label))
        outfile.write('\n')

        for q in questions:

            for key_ in ['obs', 'sim']:

                if key_ == 'obs':
                    label = 'Observed'
                elif key_ == 'sim':
                    label = 'Simulated'

                info = [label, q] + stats[key_][q]

                for i in range(len(info)):
                    if pd.isnull(info[i]):
                        info[i] = '{:>15}'.format('---')
                        continue

                    if i in [1, 2, 3]:
                        info[i] = '{:d}'.format(int(info[i]))

                    if i in [4, 5, 6, 7, 8, 9, 10]:
                        info[i] = '{:15.5f}'.format(info[i])

                outfile.write(string.format(*info))

            outfile.write('\n')

        # We calculate the RMSE based on all mean compensations.
        np_stats = np.tile(np.nan, (len(questions), 2))
        for i, q in enumerate(questions):
            for j, label in enumerate(['obs', 'sim']):
                np_stats[i, j] = stats[label][q][2]
        np_stats = np_stats[~np.isnan(np_stats).any(axis=1)]

        rmse = get_rmse(*np_stats.T)
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
