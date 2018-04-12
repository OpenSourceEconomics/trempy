from functools import partial
import string
import copy

from scipy.stats import norm
from scipy import optimize
import numpy as np

from trempy.shared.shared_constants import HUGE_FLOAT
from trempy.shared.shared_constants import TINY_FLOAT
from trempy.config_trempy import NEVER_SWITCHERS
from trempy.config_trempy import DEFAULT_BOUNDS


def criterion_function(df, questions, cutoffs, *args):
    """This function calculates the likelihood of the observed sample."""
    # Distribute parameters
    alpha, beta, eta = args[:3]
    sds = args[3:]


    m_optimal = get_optimal_compensations(questions, alpha, beta, eta)

    contribs = []

    for i, q in enumerate(questions):
        df_subset = df.loc[(slice(None), q), "Compensation"].copy().to_frame('Compensation')
        lower_cutoff, upper_cutoff = cutoffs[q]

        # TODO: This is way to clumsy.
        df_subset['is_not'] = df_subset['Compensation'].between(lower_cutoff, NEVER_SWITCHERS,
            inclusive=False)
        df_subset['is_upper'] = df_subset['Compensation'] == NEVER_SWITCHERS
        df_subset['is_lower'] = df_subset['Compensation'] == lower_cutoff

        rv = norm(loc=0.00, scale=sds[i])
        m_subset = np.repeat(m_optimal[q], sum(df_subset['is_not']), axis=0)

        # Calculate likelihoods
        arg = df_subset['Compensation'][df_subset['is_not']] - m_subset

        df_subset['likl_not'] = np.nan
        df_subset['likl_not'] = df_subset['likl_not'].mask(df_subset['is_not'] == False)

        df_subset['likl_not'].loc[df_subset['is_not']] = rv.pdf(arg)
        df_subset['likl_upper'] = 1.0 - rv.cdf(upper_cutoff)
        df_subset['likl_lower'] = rv.cdf(lower_cutoff)

        df_subset['likl'] = 0.0
        df_subset['likl'][df_subset['is_upper']] = df_subset['likl_upper'][df_subset['is_upper']]

        df_subset['likl'][df_subset['is_lower']] = df_subset['likl_lower'][df_subset['is_lower']]

        df_subset['likl'][df_subset['is_not']] = df_subset['likl_not'][df_subset['is_not']]
        contribs += df_subset['likl'].values.tolist()

    rslt = -np.mean(np.log(np.clip(sorted(contribs), TINY_FLOAT, np.inf)))

    return rslt


def get_optimal_compensations(questions, alpha, beta, eta):
    """This function returns the optimal compensations for all questions."""
    m_optimal = dict()
    for q in questions:
        m_optimal[q] = determine_optimal_compensation(alpha, beta, eta, q)
    return m_optimal


def print_init_dict(dict_, fname='test.trempy.ini'):
    """This function prints an initialization dictionary."""
    keys = []
    keys += ['PREFERENCES', 'QUESTIONS', 'CUTOFFS', 'SIMULATION', 'ESTIMATION']
    keys += ['SCIPY-BFGS', 'SCIPY-POWELL']

    questions = list(dict_['QUESTIONS'].keys())

    with open(fname, 'w') as outfile:
        for key_ in keys:

            # We do not ned to print the CUTOFFS block if none are specified.
            if key_ in ['CUTOFFS']:
                if len(dict_['CUTOFFS']) == 0:
                    continue
            outfile.write(key_ + '\n\n')
            for label in sorted(dict_[key_].keys()):
                info = dict_[key_][label]

                str_ = '{:<10}'
                if label in ['alpha', 'beta', 'eta'] + questions:
                    str_ += ' {:25.4f} {:>5} '
                else:
                    str_ += ' {:>25}\n'

                if label in ['detailed']:
                    info = str(info)

                if label in ['alpha', 'beta', 'eta'] + questions and key_ != 'CUTOFFS':
                    line, str_ = format_coefficient_line(label, info, str_)
                elif key_ in ['CUTOFFS']:
                    line, str_ = format_cutoff_line(label, info)
                    # We do not need to print a [NONE, None] cutoff.
                    if line.count('None') == 2:
                        continue

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
        if abs(cutoffs[i]) > HUGE_FLOAT:
            cutoff = 'None'
            str_ += '{:>25}'
        else:
            cutoff = np.round(cutoffs[i], decimals=4)
            str_ += '{:25.4f}'
        line += [cutoff]

    str_ += '\n'

    return line, str_


def format_coefficient_line(label, info, str_):
    """This function returns a properly formatted coefficient line."""
    value, is_fixed, bounds = info

    # We need to make sure this is an independent copy as otherwise the bound in the original
    # dictionary are overwritten with the value None.
    bounds = copy.deepcopy(bounds)

    line = []
    line += [label, value]

    if is_fixed is True:
        line += ['!']
    else:
        line += ['']

    # Bounds might be printed or now.
    for i in range(2):
        value = bounds[i]
        if value == DEFAULT_BOUNDS[label][i]:
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


def single_attribute_utility(alpha, x):
    """This function calculates the state utility."""
    return (x ** (1 - alpha)) / (1 - alpha)


def multiattribute_utility(alpha, beta, eta, x, y):
    """This function calculates the multiattribute utility."""
    u_x = single_attribute_utility(alpha, x)
    u_y = single_attribute_utility(alpha, y)
    return ((beta * u_x + (1 - beta) * u_y) ** (1 - eta)) / (1 - eta)


def expected_utility_a(alpha, beta, eta, lottery):
    """This function calculates the expected utility for lottery A."""
    if lottery == 1:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 15, 0) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 20, 0)
    elif lottery == 2:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 30, 0) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 40, 0)
    elif lottery == 3:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 60, 0) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 80, 0)
    elif lottery == 4:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 0, 15) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 0, 20)
    elif lottery == 5:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 0, 30) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 0, 40)
    elif lottery == 6:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 0, 60) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 0, 80)
    elif lottery == 7:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 15, 25) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 25, 15)
    elif lottery == 8:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 30, 50) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 50, 30)
    elif lottery == 9:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 60, 100) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 100, 60)
    elif lottery == 10:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 30, 0) + \
               0.50 * (0.50 * multiattribute_utility(alpha, beta, eta, 54, 0) +
                       0.50 * multiattribute_utility(alpha, beta, eta, 26, 0))
    elif lottery == 11:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 30, 0) + \
               0.50 * (0.80 * multiattribute_utility(alpha, beta, eta, 47, 0) +
                       0.20 * multiattribute_utility(alpha, beta, eta, 12, 0))
    elif lottery == 12:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 30, 0) + \
               0.50 * (0.80 * multiattribute_utility(alpha, beta, eta, 33, 0) +
                       0.20 * multiattribute_utility(alpha, beta, eta, 68, 0))
    elif lottery == 13:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 0, 30) + \
               0.50 * (0.50 * multiattribute_utility(alpha, beta, eta, 0, 54) +
                       0.50 * multiattribute_utility(alpha, beta, eta, 0, 26))
    elif lottery == 14:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 0, 30) + \
               0.50 * (0.80 * multiattribute_utility(alpha, beta, eta, 0, 47) +
                       0.20 * multiattribute_utility(alpha, beta, eta, 0, 12))
    elif lottery == 15:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 0, 30) + \
               0.50 * (0.80 * multiattribute_utility(alpha, beta, eta, 0, 33) +
                       0.20 * multiattribute_utility(alpha, beta, eta, 0, 68))
    else:
        raise AssertionError

    return rslt


def expected_utility_b(alpha, beta, eta, lottery, m):
    """This function calculates the expected utility for lottery B."""
    if lottery == 1:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 10 + m, 0) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 25 + m, 0)
    elif lottery == 2:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 20 + m, 0) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 50 + m, 0)
    elif lottery == 3:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 40 + m, 0) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 100 + m, 0)
    elif lottery == 4:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 0, 10 + m) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 0, 25 + m)
    elif lottery == 5:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 0, 20 + m) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 0, 50 + m)
    elif lottery == 6:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 0, 40 + m) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 0, 100 + m)
    elif lottery == 7:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 15 + m, 15) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 25 + m, 25)
    elif lottery == 8:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 30 + m, 30) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 50 + m, 50)
    elif lottery == 9:
        rslt = 0.50 * multiattribute_utility(alpha, beta, eta, 60 + m, 60) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 100 + m, 100)
    elif lottery == 10:
        rslt = 0.50 * (0.50 * multiattribute_utility(alpha, beta, eta, 44 + m, 0) +
                       0.50 * multiattribute_utility(alpha, beta, eta, 16 + m, 0)) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 40 + m, 0)
    elif lottery == 11:
        rslt = 0.50 * (0.80 * multiattribute_utility(alpha, beta, eta, 37 + m, 0) +
                       0.20 * multiattribute_utility(alpha, beta, eta, 2 + m, 0)) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 40 + m, 0)
    elif lottery == 12:
        rslt = 0.50 * (0.80 * multiattribute_utility(alpha, beta, eta, 23 + m, 0) +
                       0.20 * multiattribute_utility(alpha, beta, eta, 58 + m, 0)) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 40 + m, 0)
    elif lottery == 13:
        rslt = 0.50 * (0.50 * multiattribute_utility(alpha, beta, eta, 0, 44 + m) +
                       0.50 * multiattribute_utility(alpha, beta, eta, 0, 16 + m)) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 0, 40 + m)
    elif lottery == 14:
        rslt = 0.50 * (0.80 * multiattribute_utility(alpha, beta, eta, 0, 37 + m) +
                       0.20 * multiattribute_utility(alpha, beta, eta, 0, 2 + m)) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 0, 40 + m)
    elif lottery == 15:
        rslt = 0.50 * (0.80 * multiattribute_utility(alpha, beta, eta, 0, 23 + m) +
                       0.20 * multiattribute_utility(alpha, beta, eta, 0, 58 + m)) + \
               0.50 * multiattribute_utility(alpha, beta, eta, 0, 40 + m)
    else:
        raise AssertionError

    return rslt


def determine_optimal_compensation(alpha, beta, eta, lottery):
    """This function determine the optimal compensation that ensures the equality of the expected
    utilities."""
    def comp_criterion_function(alpha, beta, eta, lottery, m):
        """Criterion function for the root-finding function."""
        stat_a = expected_utility_a(alpha, beta, eta, lottery)
        stat_b = expected_utility_b(alpha, beta, eta, lottery, m)
        return stat_a - stat_b

    crit_func = partial(comp_criterion_function, alpha, beta, eta, lottery)

    rslt = optimize.brenth(crit_func, 0, 100)

    return rslt


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
    str_ = ''.join(np.random.choice(chars) for x in range(size))
    return str_
