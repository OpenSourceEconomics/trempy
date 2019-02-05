"""This module contains function solely related to the estimation of the model."""
import shutil
import copy
import os

from statsmodels.tools.eval_measures import rmse as get_rmse
from scipy.optimize import minimize
import pandas as pd
import numpy as np

from trempy.shared.shared_auxiliary import get_optimal_compensations
from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.shared.shared_auxiliary import char_floats
from trempy.config_trempy import PREFERENCE_PARAMETERS
from trempy.config_trempy import NEVER_SWITCHERS
from trempy.custom_exceptions import MaxfunError
from trempy.simulate.simulate import simulate
from trempy.config_trempy import SMALL_FLOAT
from trempy.config_trempy import HUGE_FLOAT
from trempy.shared.clsBase import BaseCls


class StartClass(BaseCls):
    """This class manages all issues about the model estimation."""

    def __init__(self, questions, m_optimal_obs, start_fixed,
                 start_utility_paras, version, **version_specific):
        """Init class."""
        self.attr = dict()
        self.attr['version'] = version

        # Handle version-specific objects
        if version in ['scaled_archimedean']:
            # assert all(x in version_specific.keys() for x in ['marginals', 'upper'])
            self.attr['marginals'] = version_specific['marginals']
            self.attr['upper'] = version_specific['upper']
        elif version in ['nonstationary']:
            pass

        # Initialization attributes
        self.attr['start_utility_paras'] = start_utility_paras
        self.attr['m_optimal_obs'] = m_optimal_obs
        self.attr['start_fixed'] = start_fixed
        self.attr['questions'] = questions

        # Housekeeping attributes
        self.attr['f_current'] = HUGE_FLOAT
        self.attr['f_start'] = HUGE_FLOAT
        self.attr['f_step'] = HUGE_FLOAT

        self.attr['num_eval'] = 0

    def evaluate(self, x_vals):
        """Evalute. This will be the criterion function."""
        if self.attr['num_eval'] > 10:
            return HUGE_FLOAT

        version = self.attr['version']

        if version in ['scaled_archimedean']:
            marginals = self.attr['marginals']
            upper = self.attr['upper']
            version_specific = {'upper': upper, 'marginals': marginals}
        elif version in ['nonstationary']:
            version_specific = dict()

        start_utility_paras = self.attr['start_utility_paras']
        m_optimal_obs = self.attr['m_optimal_obs']
        start_fixed = self.attr['start_fixed']
        questions = self.attr['questions']

        # Override non-fixed values in the para_obj with the xvals.
        utility_cand = copy.deepcopy(start_utility_paras)
        para_obj = utility_cand.attr['para_objs']
        j = 0
        nparas_econ = start_utility_paras.attr['nparas_econ']

        for i in range(nparas_econ):
            if start_fixed[i]:
                continue
            else:
                para_obj[i].attr['value'] = x_vals[j]
                j += 1

        # Update para_obj in utility candidate
        utility_cand.attr['para_objs'] = para_obj

        m_optimal_cand = get_optimal_compensations(
            version=version, paras_obj=utility_cand,
            questions=questions, **version_specific)
        m_optimal_cand = np.array([m_optimal_cand[q] for q in questions])

        # We need to ensure that we only compare values if the mean is not missing.
        np_stats = np.tile(np.nan, (len(questions), 2))
        for i, _ in enumerate(questions):
            np_stats[i, :] = [m_optimal_obs[i], m_optimal_cand[i]]
        np_stats = np_stats[~np.isnan(np_stats).any(axis=1)]

        fval = np.mean((np_stats[:, 0] - np_stats[:, 1]) ** 2)

        # Update class attributes
        self.attr['num_eval'] += 1

        self._update_evaluation(fval, x_vals)

        return fval

    def _update_evaluation(self, fval, x_vals):
        """Update all attributes based on the new evaluation and write some information to files."""
        self.attr['f_current'] = fval
        self.attr['num_eval'] += 1

        # Determine special events
        is_start = self.attr['num_eval'] == 1
        is_step = fval < self.attr['f_step']

        # Record information at start
        if is_start:
            self.attr['x_vals_start'] = x_vals
            self.attr['f_start'] = fval

        # Record information at step
        if is_step:
            self.attr['x_vals_step'] = x_vals
            self.attr['f_step'] = fval

        if self.attr['num_eval'] == 100:
            raise MaxfunError


def get_automatic_starting_values(paras_obj, df_obs, questions, version, **version_specific):
    """Update the container for the parameters with the automatic starting values."""
    def _adjust_bounds(value, bounds):
        """Adjust the starting values to meet the requirements of the bounds."""
        lower, upper = bounds
        if value <= bounds[0]:
            value = lower + 2 * SMALL_FLOAT
        elif value >= bounds[1]:
            value = upper - 2 * SMALL_FLOAT
        else:
            pass

        return value

    # During testing it might occur that we in fact run an estimation on a dataset that does not
    # contain any interior observations for any question. This results in a failure of the
    # automatic determination of the starting values and is thus ruled out here from the
    # beginning. In that case, we simply use the starting values from the initialization file.
    cond = df_obs['Compensation'].isin([NEVER_SWITCHERS])
    df_mask = df_obs['Compensation'].mask(cond)
    if df_mask.isnull().all():
        return paras_obj

    # We first get the observed average compensation from the data.
    m_optimal_obs = [df_mask.loc[slice(None), q].mean() for q in questions]
    m_optimal_obs = np.array(m_optimal_obs)

    # Now we gather information about the utility parameters and prepare for the interface to the
    # optimization algorithm.

    # start_utility_paras = paras_obj.get_values('econ', 'all')[:5
    start_paras, start_bounds, start_fixed = [], [], []
    for label in PREFERENCE_PARAMETERS[version]:
        value, is_fixed, bounds = paras_obj.get_para(label)
        start_fixed += [is_fixed]

        # Get list of labels that are not fixed
        if is_fixed:
            continue
        start_paras += [value]
        start_bounds += [bounds]

    # We minimize the squared distance between the observed and theoretical average
    # compensations. This is only a valid request if there are any free preference parameters.
    if len(start_paras) > 0:
        args = [questions, m_optimal_obs, start_fixed, copy.deepcopy(paras_obj), version]
        start_obj = StartClass(*args, **version_specific)

        try:
            minimize(start_obj.evaluate, start_paras, method='L-BFGS-B', bounds=start_bounds)
        except MaxfunError:
            pass
        start_utility = start_obj.get_attr('x_vals_step').tolist()

    # We construct the relevant set of free economic starting values.
    x_econ_free_start = []
    for label in PREFERENCE_PARAMETERS[version] + questions:
        value, is_fixed, bounds = paras_obj.get_para(label)

        if is_fixed:
            continue
        else:
            if label in PREFERENCE_PARAMETERS[version]:
                x_econ_free_start += [_adjust_bounds(start_utility.pop(0), bounds)]
            else:
                cond = df_obs['Compensation'].isin([NEVER_SWITCHERS])
                value = df_obs['Compensation'].mask(cond).loc[slice(None), label].std()
                # If there are no individuals observed without truncation for a particular
                # question, we start with 0.1.
                if pd.isnull(value):
                    x_econ_free_start += [_adjust_bounds(0.1, bounds)]
                else:
                    x_econ_free_start += [_adjust_bounds(value, bounds)]

    paras_obj.set_values('econ', 'free', x_econ_free_start)

    return paras_obj


def estimate_cleanup():
    """Ensure that we start the estimation with a clean slate."""
    # We remove the directories that contain the simulated choice menus at the start.
    for dirname in ['start', 'stop']:
        if os.path.exists(dirname):
            shutil.rmtree(dirname)

    # We remove the information from earlier estimation runs.
    for fname in ['est.trempy.info', 'est.trempy.log', '.stop.trempy.scratch']:
        if os.path.exists(fname):
            os.remove(fname)


def estimate_simulate(which, points, model_obj, df_obs):
    """Allow the user to easily simulate samples at the beginning and the end of the estimation."""
    paras_obj, questions, version = \
        dist_class_attributes(model_obj, 'paras_obj', 'questions', 'version')

    if version in ['scaled_archimedean']:
        upper, marginals = dist_class_attributes(model_obj, 'upper', 'marginals')
        version_specific = {'upper': upper, 'marginals': marginals}
    elif version in ['nonstationary']:
        version_specific = dict()

    m_optimal = get_optimal_compensations(version, paras_obj, questions, **version_specific)

    os.mkdir(which)
    os.chdir(which)

    sim_model = copy.deepcopy(model_obj)
    sim_model.attr['sim_file'] = which

    sim_model.update('optim', 'free', points)
    sim_model.write_out(which + '.trempy.ini')
    simulate(which + '.trempy.ini')

    compare_datasets(which, df_obs, questions, m_optimal)

    os.chdir('../')


def compare_datasets(which, df_obs, questions, m_optimal):
    """Compare the estimation dataset with a simulated one using the estimated parameter vector."""
    df_sim = pd.read_pickle(which + '.trempy.pkl')

    df_sim_masked = df_sim['Compensation'].mask(df_sim['Compensation'].isin([NEVER_SWITCHERS]))
    df_obs_masked = df_obs['Compensation'].mask(df_obs['Compensation'].isin([NEVER_SWITCHERS]))

    statistic = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

    # Summary statistics -- simulated data
    n_sim = int(df_sim_masked.shape[0] / len(questions))
    sim = df_sim_masked.groupby(['Question']).describe().to_dict(orient='index')
    stats_sim = {q: [n_sim] + [val[key] for key in statistic] for q, val in sim.items()}

    # Summary statistics -- observed data
    n_obs = int(df_obs_masked.shape[0] / len(questions))
    obs = df_obs_masked.groupby(['Question']).describe().to_dict(orient='index')
    stats_obs = {q: [n_obs] + [val[key] for key in statistic] for q, val in obs.items()}

    # Collect statistics
    stats = {'sim': stats_sim, 'obs': stats_obs}

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

        # During testing it might occur that there are no interior observations for any
        # questions.
        if np_stats.size == 0:
            rmse = '---'
        else:
            rmse = '{:15.5f}\n'.format(get_rmse(*np_stats.T))

        line = '{:>15}'.format('RMSE') + '{:>15}\n'.format(rmse)
        outfile.write(line)

        fmt_ = ' {:>10}    ' + '{:>25}    ' * 3
        for identifier, df_individual in df_obs['Compensation'].groupby(level=0):
            outfile.write('\n Individual {:d}\n\n'.format(identifier))
            outfile.write(fmt_.format(*['Question', 'Optimal', 'Observed', 'Difference']) + '\n\n')

            for (_, q), m_obs in df_individual.iteritems():
                m_opt = m_optimal[q]

                info = ['{:d}'.format(q)] + char_floats(m_opt) + char_floats(m_obs)
                info += char_floats(m_obs - m_opt)

                outfile.write(fmt_.format(*info) + '\n')
