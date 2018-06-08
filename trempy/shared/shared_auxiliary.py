"""This module contains functions that are used throughout the package."""
from functools import partial
import string
import copy

from scipy.stats import norm
from scipy import optimize
import numpy as np

from trempy.interface.interface_copulpy import get_copula
from trempy.config_trempy import PREFERENCE_PARAMETERS
from trempy.config_trempy import NEVER_SWITCHERS
from trempy.config_trempy import DEFAULT_BOUNDS
from trempy.record.clsLogger import logger_obj
from trempy.config_trempy import TINY_FLOAT
from trempy.config_trempy import HUGE_FLOAT


def criterion_function(df, questions, cutoffs, upper, *args):
    """This function calculates the likelihood of the observed sample."""
    # Distribute parameters
    r_self, r_other, delta, self, other = args[:5]
    sds = args[5:]

    m_optimal = get_optimal_compensations(questions, upper, r_self, r_other, delta, self, other)

    contribs = []
    for i, q in enumerate(questions):
        df_subset = df.loc[(slice(None), q), "Compensation"].copy().to_frame('Compensation')
        lower, upper = cutoffs[q]

        is_not = df_subset['Compensation'].between(lower, upper, inclusive=False)
        is_upper = df_subset['Compensation'].isin([NEVER_SWITCHERS])
        is_lower = df_subset['Compensation'].isin([lower])

        rv = norm(loc=m_optimal[q], scale=sds[i])

        # Calculate likelihoods
        df_subset['likl_not'] = np.nan
        df_subset['likl_not'] = df_subset['likl_not'].mask(~is_not)

        df_subset['likl_not'].loc[is_not, :] = rv.pdf(df_subset['Compensation'].loc[is_not, :])
        df_subset['likl_upper'] = 1.0 - rv.cdf(upper)
        df_subset['likl_lower'] = rv.cdf(lower)

        df_subset['likl'] = 0.0
        df_subset['likl'][is_upper] = df_subset['likl_upper'].loc[is_upper]
        df_subset['likl'][is_lower] = df_subset['likl_lower'].loc[is_lower]
        df_subset['likl'][is_not] = df_subset['likl_not'].loc[is_not]

        contribs += df_subset['likl'].values.tolist()

    rslt = -np.mean(np.log(np.clip(sorted(contribs), TINY_FLOAT, np.inf)))

    return rslt


def get_optimal_compensations(questions, upper, r_self, r_other, delta, self, other):
    """This function returns the optimal compensations for all questions."""
    copula = get_copula(upper, r_self, r_other, delta, self, other)

    m_optimal = dict()
    for q in questions:
        m_optimal[q] = determine_optimal_compensation(copula, q)
    return m_optimal


def print_init_dict(dict_, fname='test.trempy.ini'):
    """This function prints an initialization dictionary."""
    keys = []
    keys += ['UNIATTRIBUTE SELF', 'UNIATTRIBUTE OTHER', 'MULTIATTRIBUTE COPULA', 'QUESTIONS']
    keys += ['CUTOFFS', 'SIMULATION', 'ESTIMATION', 'SCIPY-BFGS', 'SCIPY-POWELL']

    questions = list(dict_['QUESTIONS'].keys())
    is_cutoffs = False

    with open(fname, 'w') as outfile:
        for key_ in keys:

            # We do not ned to print the CUTOFFS block if none are specified. So we first check
            # below if there is any need.
            if key_ not in ['CUTOFFS']:
                outfile.write(key_ + '\n\n')

            for label in sorted(dict_[key_].keys()):
                info = dict_[key_][label]

                label_internal = label
                if label in ['r'] and 'SELF' in key_:
                    label_internal = 'r_self'
                elif label in ['r'] and 'OTHER' in key_:
                    label_internal = 'r_other'

                str_ = '{:<10}'
                if label_internal in PREFERENCE_PARAMETERS + questions:
                    str_ += ' {:25.4f} {:>5} '
                else:
                    str_ += ' {:>25}\n'

                if label in ['detailed']:
                    info = str(info)

                if label_internal in PREFERENCE_PARAMETERS + questions and key_ != 'CUTOFFS':
                    line, str_ = format_coefficient_line(label_internal, info, str_)
                elif key_ in ['CUTOFFS']:
                    line, str_ = format_cutoff_line(label, info)
                    # We do not need to print a [NONE, None] cutoff.
                    if line.count('None') == 2:
                        continue
                    if not is_cutoffs:
                        is_cutoffs = True
                        outfile.write(key_ + '\n\n')

                else:
                    line = [label, info]

                outfile.write(str_.format(*line))

            outfile.write('\n')


def format_cutoff_line(label, info):
    """This function returns a properly formatted cutoff line."""
    cutoffs = info

    str_ = '{:<10}'
    line = [label]
    for i in range(2):
        if abs(cutoffs[i]) >= HUGE_FLOAT:
            cutoff = 'None'
            str_ += '{:>25}'
        else:
            cutoff = np.round(cutoffs[i], decimals=4)
            str_ += '{:25.4f}'
        line += [cutoff]

    str_ += '\n'

    return line, str_


def format_coefficient_line(label_internal, info, str_):
    """This function returns a properly formatted coefficient line."""
    value, is_fixed, bounds = info

    # We need to make sure this is an independent copy as otherwise the bound in the original
    # dictionary are overwritten with the value None.
    bounds = copy.deepcopy(bounds)

    # We need to clean up the labels for better readability.
    label_external = label_internal
    if label_internal in ['r_other', 'r_self']:
        label_external = 'r'

    line = []
    line += [label_external, value]

    if is_fixed is True:
        line += ['!']
    else:
        line += ['']

    # Bounds might be printed or now.
    for i in range(2):
        value = bounds[i]
        if value == DEFAULT_BOUNDS[label_internal][i]:
            bounds[i] = None
        else:
            bounds[i] = np.round(value, decimals=4)

    if bounds.count(None) == 2:
        bounds = ['', '']
        str_ += '{:}\n'
    else:
        str_ += '({:},{:})\n'

    line += bounds

    return line, str_


def expected_utility_a(copula, lottery):
    """This function calculates the expected utility for lottery A."""
    if lottery == 13:
        rslt = copula.evaluate(50, 0)
    elif lottery == 31:
        rslt = 0.50 * copula.evaluate(15, 0) + 0.50 * copula.evaluate(20, 0)
    elif lottery == 32:
        rslt = 0.50 * copula.evaluate(30, 0) + 0.50 * copula.evaluate(40, 0)
    elif lottery == 33:
        rslt = 0.50 * copula.evaluate(60, 0) + 0.50 * copula.evaluate(80, 0)
    elif lottery == 34:
        rslt = 0.50 * copula.evaluate(0, 15) + 0.50 * copula.evaluate(0, 20)
    elif lottery == 35:
        rslt = 0.50 * copula.evaluate(0, 30) + 0.50 * copula.evaluate(0, 40)
    elif lottery == 36:
        rslt = 0.50 * copula.evaluate(0, 60) + 0.50 * copula.evaluate(0, 80)
    elif lottery == 37:
        rslt = 0.50 * copula.evaluate(15, 25) + 0.50 * copula.evaluate(25, 15)
    elif lottery == 38:
        rslt = 0.50 * copula.evaluate(30, 50) + 0.50 * copula.evaluate(50, 30)
    elif lottery == 39:
        rslt = 0.50 * copula.evaluate(60, 100) + 0.50 * copula.evaluate(100, 60)
    elif lottery == 40:
        rslt = 0.50 * copula.evaluate(30, 0) + \
               0.50 * (0.50 * copula.evaluate(54, 0) + 0.50 * copula.evaluate(26, 0))
    elif lottery == 41:
        rslt = 0.50 * copula.evaluate(30, 0) + \
               0.50 * (0.80 * copula.evaluate(33, 0) + 0.20 * copula.evaluate(68, 0))
    elif lottery == 42:
        rslt = 0.50 * copula.evaluate(30, 0) + \
               0.50 * (0.80 * copula.evaluate(47, 0) + 0.20 * copula.evaluate(12, 0))
    elif lottery == 43:
        rslt = 0.50 * copula.evaluate(0, 30) + \
               0.50 * (0.50 * copula.evaluate(0, 54) + 0.50 * copula.evaluate(0, 26))
    elif lottery == 44:
        rslt = 0.50 * copula.evaluate(0, 30) + \
               0.50 * (0.80 * copula.evaluate(0, 33) + 0.20 * copula.evaluate(0, 68))
    elif lottery == 45:
        rslt = 0.50 * copula.evaluate(0, 30) + \
               0.50 * (0.80 * copula.evaluate(0, 47) + 0.20 * copula.evaluate(0, 12))
    else:
        raise AssertionError

    return rslt


def expected_utility_b(copula, lottery, m):
    """This function calculates the expected utility for lottery B."""
    if lottery == 13:
        rslt = copula.evaluate(0, m)
    elif lottery == 31:
        rslt = 0.50 * copula.evaluate(10 + m, 0) + \
               0.50 * copula.evaluate(25 + m, 0)
    elif lottery == 32:
        rslt = 0.50 * copula.evaluate(20 + m, 0) + \
               0.50 * copula.evaluate(50 + m, 0)
    elif lottery == 33:
        rslt = 0.50 * copula.evaluate(40 + m, 0) + \
               0.50 * copula.evaluate(100 + m, 0)
    elif lottery == 34:
        rslt = 0.50 * copula.evaluate(0, 10 + m) + \
               0.50 * copula.evaluate(0, 25 + m)
    elif lottery == 35:
        rslt = 0.50 * copula.evaluate(0, 20 + m) + \
               0.50 * copula.evaluate(0, 50 + m)
    elif lottery == 36:
        rslt = 0.50 * copula.evaluate(0, 40 + m) + \
               0.50 * copula.evaluate(0, 100 + m)
    elif lottery == 37:
        rslt = 0.50 * copula.evaluate(15 + m, 15) + \
               0.50 * copula.evaluate(25 + m, 25)
    elif lottery == 38:
        rslt = 0.50 * copula.evaluate(30 + m, 30) + \
               0.50 * copula.evaluate(50 + m, 50)
    elif lottery == 39:
        rslt = 0.50 * copula.evaluate(60 + m, 60) + \
               0.50 * copula.evaluate(100 + m, 100)
    elif lottery == 40:
        rslt = 0.50 * (0.50 * copula.evaluate(44 + m, 0) +
                       0.50 * copula.evaluate(16 + m, 0)) + \
               0.50 * copula.evaluate(40 + m, 0)
    elif lottery == 41:
        rslt = 0.50 * (0.80 * copula.evaluate(23 + m, 0) +
                       0.20 * copula.evaluate(58 + m, 0)) + \
               0.50 * copula.evaluate(40 + m, 0)
    elif lottery == 42:
        rslt = 0.50 * (0.80 * copula.evaluate(37 + m, 0) +
                       0.20 * copula.evaluate(2 + m, 0)) + \
               0.50 * copula.evaluate(40 + m, 0)
    elif lottery == 43:
        rslt = 0.50 * (0.50 * copula.evaluate(0, 44 + m) +
                       0.50 * copula.evaluate(0, 16 + m)) + \
               0.50 * copula.evaluate(0, 40 + m)
    elif lottery == 44:
        rslt = 0.50 * (0.80 * copula.evaluate(0, 23 + m) +
                       0.20 * copula.evaluate(0, 58 + m)) + \
               0.50 * copula.evaluate(0, 40 + m)
    elif lottery == 45:
        rslt = 0.50 * (0.80 * copula.evaluate(0, 37 + m) +
                       0.20 * copula.evaluate(0, 2 + m)) + \
               0.50 * copula.evaluate(0, 40 + m)
    else:
        raise AssertionError

    return rslt


def determine_optimal_compensation(copula, lottery):
    """This function determines the optimal compensation that ensures the equality of teh
    expected utilities."""
    def comp_criterion_function(copula, lottery, version, m):
        """Criterion function for the root-finding function."""
        stat_a = expected_utility_a(copula, lottery)
        stat_b = expected_utility_b(copula, lottery, m)

        if version == 'brenth':
            stat = stat_a - stat_b
        elif version == 'grid':
            stat = (stat_a - stat_b) ** 2
        else:
            raise NotImplementedError
        return stat

    # For some parametrization our first choice fails as f(a) and f(b) must have different
    # signs. If that is the case, we use a simple grid search as backup.
    try:
        crit_func = partial(comp_criterion_function, copula, lottery, 'brenth')
        m_opt = optimize.brenth(crit_func, 0.00, 200)
    except ValueError:
        crit_func = partial(comp_criterion_function, copula, lottery, 'grid')
        crit_func = np.vectorize(crit_func)
        grid = np.linspace(0, 200, num=500, endpoint=True)
        m_opt = grid[np.argmin(crit_func(grid))]
        logger_obj.record_event(2)

    return m_opt


def dist_class_attributes(model_obj, *args):
    """ This function distributes a host of class attributes.
    """
    # Initialize container
    ret = []

    # Process requests
    for arg in args:
        ret.append(model_obj.get_attr(arg))

    # There is some special handling for the case where only one element is returned.
    if len(ret) == 1:
        ret = ret[0]

    # Finishing
    return ret


def get_random_string(size=6):
    """This function samples a random string of varying size."""
    chars = list(string.ascii_lowercase)
    str_ = ''.join(np.random.choice(chars) for _ in range(size))
    return str_
