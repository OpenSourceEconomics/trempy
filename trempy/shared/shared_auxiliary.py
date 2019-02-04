"""This module contains functions that are used throughout the package."""
from functools import partial
import string
import copy

from scipy.stats import norm
from scipy import optimize
import pandas as pd
import numpy as np

from trempy.interface.interface_copulpy import get_copula_nonstationary
from trempy.interface.interface_copulpy import get_copula_scaled_archimedean
from trempy.custom_exceptions import TrempyError
from trempy.config_trempy import PREFERENCE_PARAMETERS
from trempy.config_trempy import NEVER_SWITCHERS
from trempy.config_trempy import DEFAULT_BOUNDS
from trempy.config_trempy import LOTTERY_BOUNDS
from trempy.config_trempy import TINY_FLOAT
from trempy.config_trempy import HUGE_FLOAT


def criterion_function(df, questions, cutoffs, paras_obj, version, sds, **version_specific):
    """Calculate the likelihood of the observed sample."""
    m_optimal = get_optimal_compensations(
        version=version, paras_obj=paras_obj, questions=questions, **version_specific)
    data = copy.deepcopy(df)

    # Add auxiliary data (question cutoffs, decision implied by the model, std of observed choices)
    df_cutoff = pd.DataFrame.from_dict(cutoffs, orient='index', columns=['lower', 'upper'])
    df_cutoff.index.name = 'Question'
    data = data.join(df_cutoff, how='left')

    df_m_optimal = pd.DataFrame.from_dict(m_optimal, orient='index', columns=['m_optim'])
    df_m_optimal.index.name = 'Question'
    data = data.join(df_m_optimal, how='left')

    df_std = pd.DataFrame(sds, index=questions, columns=['std'])
    df_std.index.name = 'Question'
    data = data.join(df_std, how='left')

    # Subjects who selected both Option A and B at least once. This implies their valuation
    # is in the left-open interval (lower, upper], i.e. they initially prefered Option A at 'lower',
    # but chose option B when it offered 'upper'.
    data['is_interior'] = (data.lower < data.Compensation) & (data.Compensation < data.upper)
    # Subjects who always prefered option A, i.e. their value of option A is higher than 'upper'.
    data['is_upper'] = ((data['Compensation'].isin([NEVER_SWITCHERS])) |
                        (data.Compensation > data.upper))
    # Subjects who always prefered option B. So their value of Option A is smaller than 'lower'
    data['is_lower'] = (data.Compensation <= data.lower)

    # We only need the standard normal distribution for standardized choices.
    data['choice_standardized'] = (data['Compensation'] - data['m_optim']) / data['std']
    data['lower_standardized'] = (data['lower'] - data['m_optim']) / data['std']
    data['upper_standardized'] = (data['upper'] - data['m_optim']) / data['std']
    rv = norm(loc=0.0, scale=1.0)

    # Likelihood: pdf for interior choices
    likl_interior = (rv.pdf(data['choice_standardized'].loc[data['is_interior']]) /
                     data['std'].loc[data['is_interior']])

    # Likelihood: cdf for indifference points that are outside our choice list.
    likl_upper = 1.0 - rv.cdf(data['upper_standardized'].loc[data['is_upper']])
    likl_lower = rv.cdf(data['lower_standardized'].loc[data['is_lower']])

    # Average negative log-likelihood
    contribs = likl_interior.tolist() + likl_lower.tolist() + likl_upper.tolist()
    rslt = - np.mean(np.log(np.clip(sorted(contribs), TINY_FLOAT, np.inf)))
    return rslt, m_optimal


def get_optimal_compensations_scaled_archimedean(questions, upper, marginals, r_self,
                                                 r_other, delta, self, other):
    """Return the optimal compensations for all questions."""
    for question in questions:
        if question <= 30 and not question == 13:
            raise TrempyError('Temporal decisions not implemented for scaled_archimedean.')

    copula = get_copula_scaled_archimedean(upper, marginals, r_self, r_other, delta, self, other)

    m_optimal = dict()
    for q in questions:
        m_optimal[q] = determine_optimal_compensation(copula, q)
    return m_optimal


def get_optimal_compensations_nonstationary(questions, alpha, beta, gamma, y_scale,
                                            discount_factors_0, discount_factors_1,
                                            discount_factors_3, discount_factors_6,
                                            discount_factors_12, discount_factors_24,
                                            unrestricted_weights_0, unrestricted_weights_1,
                                            unrestricted_weights_3, unrestricted_weights_6,
                                            unrestricted_weights_12, unrestricted_weights_24,
                                            # Optional arguments that determine the model type
                                            discounting, stationary_model
                                            ):
    """Optimal compensation for the nonstationary utility function."""
    copula = get_copula_nonstationary(
        alpha, beta, gamma, y_scale,
        discount_factors_0, discount_factors_1,
        discount_factors_3, discount_factors_6,
        discount_factors_12, discount_factors_24,
        unrestricted_weights_0, unrestricted_weights_1,
        unrestricted_weights_3, unrestricted_weights_6,
        unrestricted_weights_12, unrestricted_weights_24,
        discounting=discounting,
        stationary_model=stationary_model)

    m_optimal = dict()
    for q in questions:
        m_optimal[q] = determine_optimal_compensation(copula, q)
    return m_optimal


def get_optimal_compensations(version, paras_obj, questions, **version_specific):
    """Get optimal compensations based on a model_obj."""
    nparas_econ = paras_obj.attr['nparas_econ']

    if version in ['scaled_archimedean']:
        # Handle version-specific objects outside paras_obj
        # assert 'marginals' in version_specific.keys()
        # assert 'upper' in version_specific.keys()
        marginals = version_specific['marginals']
        upper = version_specific['upper']

        # Variable args
        r_self, r_other, delta, self, other = paras_obj.get_values('econ', 'all')[:nparas_econ]

        # Optimal compensation
        args = [questions, upper, marginals, r_self, r_other, delta, self, other]
        m_optimal = get_optimal_compensations_scaled_archimedean(*args)

    elif version in ['nonstationary']:
        # Variable args
        # TODO: How to handle optional arguments? If unrestricted_weights is not generated,
        # this does not work.
        alpha, beta, gamma, y_scale, discount_factors_0, discount_factors_1, \
            discount_factors_3, discount_factors_6, discount_factors_12, discount_factors_24, \
            unrestricted_weights_0, unrestricted_weights_1, unrestricted_weights_3, \
            unrestricted_weights_6, unrestricted_weights_12, unrestricted_weights_24 = \
            paras_obj.get_values('econ', 'all')[:nparas_econ]

        # Optional arguments
        discounting = paras_obj.attr['discounting']
        stationary_model = paras_obj.attr['stationary_model']

        # Optimal compensation
        args = [questions, alpha, beta, gamma, y_scale,
                discount_factors_0, discount_factors_1,
                discount_factors_3, discount_factors_6,
                discount_factors_12, discount_factors_24,
                unrestricted_weights_0, unrestricted_weights_1, unrestricted_weights_3,
                unrestricted_weights_6, unrestricted_weights_12, unrestricted_weights_24,
                # Optional arguments:
                discounting, stationary_model]
        m_optimal = get_optimal_compensations_nonstationary(*args)
    else:
        raise TrempyError('version not implemented')

    return m_optimal


def print_init_dict(dict_, fname='test.trempy.ini'):
    """Print an initialization dictionary."""
    version = dict_['VERSION']['version']

    keys = ['VERSION', 'SIMULATION', 'ESTIMATION',
            'SCIPY-BFGS', 'SCIPY-POWELL', 'SCIPY-L-BFGS-B',
            'CUTOFFS', 'QUESTIONS']

    # Add keys based on version of the utility function
    if version in ['scaled_archimedean']:
        keys += ['UNIATTRIBUTE SELF', 'UNIATTRIBUTE OTHER', 'MULTIATTRIBUTE COPULA']
    elif version in ['nonstationary']:
        keys += ['ATEMPORAL', 'DISCOUNTING']
    else:
        raise TrempyError('version not implemented')

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

                # Manually translate labels to internal labels based on version
                if version in ['scaled_archimedean']:
                    if label in ['r'] and 'SELF' in key_:
                        label_internal = 'r_self'
                    elif label in ['r'] and 'OTHER' in key_:
                        label_internal = 'r_other'
                elif version in ['nonstationary']:
                    pass

                # Build format string for line
                str_ = '{:<25}'
                if label_internal in PREFERENCE_PARAMETERS[version] + questions:
                    # Handle optional arguments where None can occur
                    if (isinstance(label_internal, str) and
                       label_internal.startswith('unrestricted_weights') and info[0] is None):
                        str_ += ' {:>25} {:>10} '
                    # Preference parameters are formatted as floats
                    else:
                        str_ += ' {:25.4f} {:>10} '
                else:
                    # All other parameters are formatted as strings
                    str_ += ' {:>25}\n'

                # Handle string output (e.g. "True" or "None")
                if label in ['detailed', 'version']:
                    info = str(info)
                if label in ['discounting', 'stationary_model']:
                    if info is None:
                        info = 'None'
                    else:
                        info = str(info)

                if (label_internal in PREFERENCE_PARAMETERS[version] + questions and
                   key_ != 'CUTOFFS'):
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
    """Return a properly formatted cutoff line."""
    cutoffs = info

    str_ = '{:<26}'
    line = [label]
    for i in range(2):
        if abs(cutoffs[i]) >= HUGE_FLOAT:
            cutoff = 'None'
            str_ += '{:>25} '
        else:
            cutoff = np.round(cutoffs[i], decimals=4)
            str_ += '{:25.4f} '
        line += [cutoff]

    str_ += '\n'

    return line, str_


def format_coefficient_line(label_internal, info, str_):
    """Return a properly formatted coefficient line."""
    value, is_fixed, bounds = info

    # We need to make sure this is an independent copy as otherwise the bound in the original
    # dictionary are overwritten with the value None.
    bounds = copy.deepcopy(bounds)

    # We need to clean up the labels for better readability.
    label_external = label_internal
    if label_internal in ['r_other', 'r_self']:
        label_external = 'r'

    # First, filter out integer values
    if isinstance(label_external, np.int64) or isinstance(label_external, int):
        line = [label_external, value]
    # Handle optional arguments that should be set to 'None'
    elif label_external.startswith('unrestricted_weights') and value is None:
        line = [label_external, 'None']
    # Handle all other cases
    else:
        # The regular case where we want to print value as a float
        line = [label_external, value]

    if np.any(is_fixed):
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
    """Calculate the expected utility for lottery A."""
    # TEMPORAL DECISIONS
    if lottery in [1, 2, 3, 4, 5, 19, 20, 21, 22, 23]:
        rslt = copula.evaluate(50, 0, t=0)
    # Note: question 13 is temporal but t=0. So it is handled under risky choices.
    elif lottery in [7, 8, 9, 10, 11, 25, 26, 27, 28, 29]:
        rslt = copula.evaluate(0, 50, t=0)
    elif lottery in [6, 16, 24]:
        rslt = copula.evaluate(50, 0, t=6)
    elif lottery in [12, 30]:
        rslt = copula.evaluate(0, 50, t=6)
    elif lottery == 14:
        rslt = copula.evaluate(50, 0, t=1)
    elif lottery == 15:
        rslt = copula.evaluate(50, 0, t=3)
    elif lottery == 17:
        rslt = copula.evaluate(50, 0, t=12)
    elif lottery == 18:
        rslt = copula.evaluate(50, 0, t=24)

    # RISKY CHOICES
    elif lottery == 13:
        rslt = copula.evaluate(50, 0, t=0)
    elif lottery == 31:
        rslt = 0.50 * copula.evaluate(15, 0, t=0) + 0.50 * copula.evaluate(20, 0, t=0)
    elif lottery == 32:
        rslt = 0.50 * copula.evaluate(30, 0, t=0) + 0.50 * copula.evaluate(40, 0, t=0)
    elif lottery == 33:
        rslt = 0.50 * copula.evaluate(60, 0, t=0) + 0.50 * copula.evaluate(80, 0, t=0)
    elif lottery == 34:
        rslt = 0.50 * copula.evaluate(0, 15, t=0) + 0.50 * copula.evaluate(0, 20, t=0)
    elif lottery == 35:
        rslt = 0.50 * copula.evaluate(0, 30, t=0) + 0.50 * copula.evaluate(0, 40, t=0)
    elif lottery == 36:
        rslt = 0.50 * copula.evaluate(0, 60, t=0) + 0.50 * copula.evaluate(0, 80, t=0)
    elif lottery == 37:
        rslt = 0.50 * copula.evaluate(15, 25, t=0) + 0.50 * copula.evaluate(25, 15, t=0)
    elif lottery == 38:
        rslt = 0.50 * copula.evaluate(30, 50, t=0) + 0.50 * copula.evaluate(50, 30, t=0)
    elif lottery == 39:
        rslt = 0.50 * copula.evaluate(60, 100, t=0) + 0.50 * copula.evaluate(100, 60, t=0)
    elif lottery == 40:
        rslt = 0.50 * copula.evaluate(30, 0, t=0) + \
            0.50 * (0.50 * copula.evaluate(54, 0, t=0) + 0.50 * copula.evaluate(26, 0, t=0))
    elif lottery == 41:
        rslt = 0.50 * copula.evaluate(30, 0, t=0) + \
            0.50 * (0.80 * copula.evaluate(33, 0, t=0) + 0.20 * copula.evaluate(68, 0, t=0))
    elif lottery == 42:
        rslt = 0.50 * copula.evaluate(30, 0, t=0) + \
            0.50 * (0.80 * copula.evaluate(47, 0, t=0) + 0.20 * copula.evaluate(12, 0, t=0))
    elif lottery == 43:
        rslt = 0.50 * copula.evaluate(0, 30, t=0) + \
            0.50 * (0.50 * copula.evaluate(0, 54, t=0) + 0.50 * copula.evaluate(0, 26, t=0))
    elif lottery == 44:
        rslt = 0.50 * copula.evaluate(0, 30, t=0) + \
            0.50 * (0.80 * copula.evaluate(0, 33, t=0) + 0.20 * copula.evaluate(0, 68, t=0))
    elif lottery == 45:
        rslt = 0.50 * copula.evaluate(0, 30, t=0) + \
            0.50 * (0.80 * copula.evaluate(0, 47, t=0) + 0.20 * copula.evaluate(0, 12, t=0))
    else:
        raise AssertionError

    return rslt


def expected_utility_b(copula, lottery, m):
    """Calculate the expected utility for lottery B."""
    # TEMPORAL CHOICES

    # Univariate discounting: SELF. 0-1, 0-3, 0-6, 0-12, 0-24, 6-12
    if lottery == 1:
        rslt = copula.evaluate(m, 0, t=1)
    elif lottery == 2:
        rslt = copula.evaluate(m, 0, t=3)
    elif lottery == 3:
        rslt = copula.evaluate(m, 0, t=6)
    elif lottery == 4:
        rslt = copula.evaluate(m, 0, t=12)
    elif lottery == 5:
        rslt = copula.evaluate(m, 0, t=24)
    elif lottery == 6:
        rslt = copula.evaluate(m, 0, t=12)

    # Univariate discounting: CHARITY. 0-1, 0-3, 0-6, 0-12, 0-24, 6-12
    elif lottery == 7:
        rslt = copula.evaluate(0, m, t=1)
    elif lottery == 8:
        rslt = copula.evaluate(0, m, t=3)
    elif lottery == 9:
        rslt = copula.evaluate(0, m, t=6)
    elif lottery == 10:
        rslt = copula.evaluate(0, m, t=12)
    elif lottery == 11:
        rslt = copula.evaluate(0, m, t=24)
    elif lottery == 12:
        rslt = copula.evaluate(0, m, t=12)

    # Exchange rate. 0-0, 1-1, 3-3, 6-6, 12-12, 24-24
    # Question 13 is counted as a riskless lottery question because t=0.
    elif lottery == 13:
        rslt = copula.evaluate(0, m, t=0)
    elif lottery == 14:
        rslt = copula.evaluate(0, m, t=1)
    elif lottery == 15:
        rslt = copula.evaluate(0, m, t=3)
    elif lottery == 16:
        rslt = copula.evaluate(0, m, t=6)
    elif lottery == 17:
        rslt = copula.evaluate(0, m, t=12)
    elif lottery == 18:
        rslt = copula.evaluate(0, m, t=24)

    # Multivariate discounting: SELF. 0-1, 0-3, 0-6, 0-12, 0-24, 6-12
    elif lottery == 19:
        rslt = copula.evaluate(0, m, t=1)
    elif lottery == 20:
        rslt = copula.evaluate(0, m, t=3)
    elif lottery == 21:
        rslt = copula.evaluate(0, m, t=6)
    elif lottery == 22:
        rslt = copula.evaluate(0, m, t=12)
    elif lottery == 23:
        rslt = copula.evaluate(0, m, t=24)
    elif lottery == 24:
        rslt = copula.evaluate(0, m, t=12)

    # Multivariate discounting: CHARITY. 0-1, 0-3, 0-6, 0-12, 0-24, 6-12
    elif lottery == 25:
        rslt = copula.evaluate(m, 0, t=1)
    elif lottery == 26:
        rslt = copula.evaluate(m, 0, t=3)
    elif lottery == 27:
        rslt = copula.evaluate(m, 0, t=6)
    elif lottery == 28:
        rslt = copula.evaluate(m, 0, t=12)
    elif lottery == 29:
        rslt = copula.evaluate(m, 0, t=24)
    elif lottery == 30:
        rslt = copula.evaluate(m, 0, t=12)

    # RISKY CHOICES
    elif lottery == 31:
        rslt = 0.50 * copula.evaluate(10 + m, 0, t=0) + 0.50 * copula.evaluate(25 + m, 0, t=0)
    elif lottery == 32:
        rslt = 0.50 * copula.evaluate(20 + m, 0, t=0) + 0.50 * copula.evaluate(50 + m, 0, t=0)
    elif lottery == 33:
        rslt = 0.50 * copula.evaluate(40 + m, 0, t=0) + 0.50 * copula.evaluate(100 + m, 0, t=0)
    elif lottery == 34:
        rslt = 0.50 * copula.evaluate(0, 10 + m, t=0) + 0.50 * copula.evaluate(0, 25 + m, t=0)
    elif lottery == 35:
        rslt = 0.50 * copula.evaluate(0, 20 + m, t=0) + 0.50 * copula.evaluate(0, 50 + m, t=0)
    elif lottery == 36:
        rslt = 0.50 * copula.evaluate(0, 40 + m, t=0) + 0.50 * copula.evaluate(0, 100 + m, t=0)
    elif lottery == 37:
        rslt = 0.50 * copula.evaluate(15 + m, 15, t=0) + 0.50 * copula.evaluate(25 + m, 25, t=0)
    elif lottery == 38:
        rslt = 0.50 * copula.evaluate(30 + m, 30, t=0) + 0.50 * copula.evaluate(50 + m, 50, t=0)
    elif lottery == 39:
        rslt = 0.50 * copula.evaluate(60 + m, 60, t=0) + 0.50 * copula.evaluate(100 + m, 100, t=0)
    elif lottery == 40:
        rslt = 0.50 * (0.50 * copula.evaluate(44 + m, 0, t=0) + 0.50 *
                       copula.evaluate(16 + m, 0, t=0)) + 0.50 * copula.evaluate(40 + m, 0, t=0)
    elif lottery == 41:
        rslt = 0.50 * (0.80 * copula.evaluate(23 + m, 0, t=0) + 0.20 *
                       copula.evaluate(58 + m, 0, t=0)) + 0.50 * copula.evaluate(40 + m, 0, t=0)
    elif lottery == 42:
        rslt = 0.50 * (0.80 * copula.evaluate(37 + m, 0, t=0) + 0.20 *
                       copula.evaluate(2 + m, 0, t=0)) + 0.50 * copula.evaluate(40 + m, 0, t=0)
    elif lottery == 43:
        rslt = 0.50 * (0.50 * copula.evaluate(0, 44 + m, t=0) + 0.50 *
                       copula.evaluate(0, 16 + m, t=0)) + 0.50 * copula.evaluate(0, 40 + m, t=0)
    elif lottery == 44:
        rslt = 0.50 * (0.80 * copula.evaluate(0, 23 + m, t=0) + 0.20 *
                       copula.evaluate(0, 58 + m, t=0)) + 0.50 * copula.evaluate(0, 40 + m, t=0)
    elif lottery == 45:
        rslt = 0.50 * (0.80 * copula.evaluate(0, 37 + m, t=0) + 0.20 *
                       copula.evaluate(0, 2 + m, t=0)) + 0.50 * copula.evaluate(0, 40 + m, t=0)
    else:
        raise AssertionError

    return rslt


def determine_optimal_compensation(copula, lottery):
    """Determine the optimal compensation that ensures the equality of the expected utilities."""
    def comp_criterion_function(copula, lottery, m):
        """Criterion function for the root-finding function."""
        stat_a = expected_utility_a(copula, lottery)
        stat_b = expected_utility_b(copula, lottery, m)
        stat = stat_a - stat_b
        return stat

    lower, upper = LOTTERY_BOUNDS[lottery]
    crit_func = partial(comp_criterion_function, copula, lottery)

    # If the criterion function is positive even at the maximum compensation then the optimal
    # compensation is set to upper bound itself.
    if np.sign(crit_func(upper)) == 1:
        m_opt = float(upper)
    # If the criterion function is already even at the minimum compensation then the optimal
    # compensation is set to the lower bound itself.
    elif np.sign(crit_func(lower)) == -1:
        m_opt = float(lower)
    else:
        m_opt = optimize.brenth(crit_func, lower, upper)

    return m_opt


def dist_class_attributes(model_obj, *args):
    """Distribute a host of class attributes."""
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
    """Sample a random string of varying size."""
    chars = list(string.ascii_lowercase)
    str_ = ''.join(np.random.choice(chars) for _ in range(size))
    return str_


def char_floats(floats):
    """Ensure a pretty printing of all floats."""
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
