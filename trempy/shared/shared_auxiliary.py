from functools import partial
import string
import copy

from scipy.stats import truncnorm
from scipy.stats import norm
from scipy import optimize
import numpy as np

from trempy.shared.shared_constants import HUGE_FLOAT, TINY_FLOAT


def criterion_function(df, questions, cutoffs, *args):
    """This function calculates the likelihood."""
    # TODO: This treats the dataset input as the final version, a lot of information is deduced
    # from this.

    alpha, beta, eta = args[:3]
    sds = args[3:]


    repeat = []
    for q in questions:
        repeat += [df['Question'][df['Question'] == q].count()]

    m_optimal = []
    for q in questions:
        m_optimal += [determine_optimal_compensation(alpha, beta, eta, q)]

    contribs = []

    for i, q in enumerate(questions):

        df_subset = df[df['Question'] == q].copy()

        lower_cutoff, upper_cutoff = cutoffs[q]
        df_subset['is_truncated'] = df_subset['Compensation'] == lower_cutoff

        rv = norm(loc=0.00, scale=sds[i])
        rv_trunc = truncnorm(-HUGE_FLOAT, upper_cutoff, loc=0.00, scale=sds[i])

        m_subset = np.repeat(m_optimal[i], len(df_subset), axis=0)

        df_subset['likl_not_trunc'] = rv_trunc.pdf(df_subset['Compensation'] - m_subset)

        df_subset['likl_trunc'] = rv.cdf(lower_cutoff)

        contribs += (df_subset['is_truncated'] * df_subset['likl_trunc'] + (1.0 - df_subset[
            'is_truncated']) * df_subset['likl_not_trunc']).values.tolist()

    # TODO: This needs to be logged somwhere
    rslt = -np.mean(np.log(np.clip(sorted(contribs), TINY_FLOAT, np.inf)))

    return rslt


def print_init_dict(dict_, fname='test.trempy.ini'):
    """This function prints an initialization dictionary."""
    keys = ['PREFERENCES', 'QUESTIONS', 'CUTOFFS', 'SIMULATION', 'ESTIMATION']
    questions = list(dict_['QUESTIONS'].keys())

    with open(fname, 'w') as outfile:
        for key_ in keys:
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
                elif label in questions and key_ == 'CUTOFFS':
                    line, str_ = format_cutoff_line(label, info, str_)
                    # We do not need to print a [NONE, None] cutoff.
                    if line.count('None') == 2:
                        continue

                else:
                    line = [label, info]

                outfile.write(str_.format(*line))

            outfile.write('\n')


def format_cutoff_line(label, info, str_):
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


def format_question_line(label, info, str_):
    """This function returns a properly formatted question line."""
    cutoffs, value, is_fixed, bounds = info

    # We need to make sure this is an independent copy as otherwise the bound in the original
    # dictionary are overwritten with the value None.
    cutoffs = copy.deepcopy(cutoffs)

    line = []
    line += [label, value]

    if is_fixed == 'True':
        line += ['!']
    else:
        line += ['']

    # Bounds might be printed or now.
    for i in range(2):
        value = bounds[i]
        if abs(value) > HUGE_FLOAT:
            bounds[i] = None
        else:
            bounds[i] = np.round(value, decimals=4)

    for i in range(2):
        if abs(cutoffs[i]) > HUGE_FLOAT:
            cutoff = None
        else:
            cutoff = np.round(cutoffs[i], decimals=4)
        line += [cutoff]

#    if cutoffs.count(None) == 2:
#        cutoffs = ['', '']
#        str_ += '{:}\n'
#    else:
#        str_ += '({:},{:})\n'

    line += cutoffs

    return line, str_


def format_coefficient_line(label, info, str_):
    """This function returns a properly formatted coefficient line."""
    value, is_fixed, bounds = info

    # We need to make sure this is an independent copy as otherwise the bound in the original
    # dictionary are overwritten with the value None.
    bounds = copy.deepcopy(bounds)

    line = []
    line += [label, value]

    if is_fixed == 'True':
        line += ['!']
    else:
        line += ['']

    # Bounds might be printed or now.
    for i in range(2):
        value = bounds[i]
        if abs(value) > HUGE_FLOAT:
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
    def criterion_function(alpha, beta, eta, lottery, m):
        """Criterion function for the root-finding function."""
        return expected_utility_a(alpha, beta, eta, lottery) - \
               expected_utility_b(alpha, beta, eta, lottery, m)

    crit_func = partial(criterion_function, alpha, beta, eta, lottery)

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
