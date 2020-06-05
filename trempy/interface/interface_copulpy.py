"""Manage everything related to interfacing the copulpy package."""
import numpy as np

from trempy.custom_exceptions import TrempyError
from copulpy import UtilityCopulaCls


def get_copula_scaled_archimedean(upper, marginals, r_self, r_other, delta, self, other):
    """Access a multiattribute utility copula."""
    copula_spec = dict()

    version = 'scaled_archimedean'
    copula_spec['version'] = version
    copula_spec[version] = dict()

    copula_spec[version]['r'] = [r_self, r_other]
    copula_spec[version]['marginals'] = marginals
    copula_spec[version]['u'] = self, other
    copula_spec[version]['bounds'] = upper
    copula_spec[version]['delta'] = delta

    copula_spec[version]['generating_function'] = 1
    copula_spec[version]['version'] = 'scaled_archimedean'
    copula_spec[version]['a'] = 1.0
    copula_spec[version]['b'] = 0.0

    copula = UtilityCopulaCls(copula_spec)

    return copula


def get_copula_nonstationary(alpha, beta, gamma, y_scale,
                             discount_factors_0, discount_factors_1, discount_factors_3,
                             discount_factors_6, discount_factors_12, discount_factors_24,
                             unrestricted_weights_0, unrestricted_weights_1,
                             unrestricted_weights_3, unrestricted_weights_6,
                             unrestricted_weights_12, unrestricted_weights_24,
                             discounting=None,
                             stationary_model=False,
                             df_other='equal_univariate'
                             ):
    """Access the nonstationary utility copula."""
    # Anti-bugging.
    np.testing.assert_equal(discounting in [None, 'hyperbolic', 'exponential'], True)

    version = 'nonstationary'
    copula_spec = {'version': version}
    copula_spec[version] = {
        'discounting': discounting,
        'version': version,
        'y_scale': y_scale,
        'alpha': alpha,
        'gamma': gamma,
        'beta': beta,
    }

    # "Nonparametric" discount factors D_t for t in 0,1,3,6,12,24.
    dfx = {
        0: discount_factors_0,
        1: discount_factors_1,
        3: discount_factors_3,
        6: discount_factors_6,
        12: discount_factors_12,
        24: discount_factors_24,
    }
    copula_spec[version]['discount_factors'] = dfx

    if df_other in ['equal_univariate']:
        # We use the parametric restrictions on c_t derived from theory.
        dict_unrestricted = None
    elif df_other in ['free']:
        # The weight c_t in the CES function is free.
        dict_unrestricted = {
            0: unrestricted_weights_0,
            1: unrestricted_weights_1,
            3: unrestricted_weights_3,
            6: unrestricted_weights_6,
            12: unrestricted_weights_12,
            24: unrestricted_weights_24,
        }
        if None in dict_unrestricted.values():
            raise TrempyError('discount function for other is set to free but contains None type')
    elif df_other in ['linear']:
        # Impose a linear structure on c_t in the CES function.
        dict_unrestricted = {
            t: max(0, y_scale + t * unrestricted_weights_0) for t in [0, 1, 3, 6, 12, 24]
        }
    elif df_other in ['exponential']:
        # Impose an exponential structure on c_t in the CES function.
        dict_unrestricted = {t: y_scale * unrestricted_weights_0 ** t for t in [0, 1, 3, 6, 12, 24]}

    # The model becomes stationary.
    if stationary_model is True:
        dict_unrestricted = {key: y_scale for key in [0, 1, 3, 6, 12, 24]}

    copula_spec[version]['unrestricted_weights'] = dict_unrestricted

    # Build copula
    copula = UtilityCopulaCls(copula_spec)

    return copula


def get_copula_warmglow(alpha, beta, gamma, y_scale,
                        discount_factors_0, discount_factors_1, discount_factors_3,
                        discount_factors_6, discount_factors_12, discount_factors_24,
                        unrestricted_weights_0, unrestricted_weights_1,
                        unrestricted_weights_3, unrestricted_weights_6,
                        unrestricted_weights_12, unrestricted_weights_24,
                        discounting=None,
                        stationary_model=False,
                        df_other='equal_univariate',
                        warmglow_type='constant'
                        ):
    """Access the warm glow utility function."""
    # Anti-bugging.
    np.testing.assert_equal(discounting in [None, 'hyperbolic', 'exponential'], True)
    np.testing.assert_equal(warmglow_type in ['constant', 'linear'], True)

    version = 'warmglow'
    copula_spec = {'version': version}
    copula_spec[version] = {
        'discounting': discounting,
        'version': version,
        'y_scale': y_scale,
        'alpha': alpha,
        'gamma': gamma,
        'beta': beta,
        'warmglow_type': warmglow_type,
    }

    # "Nonparametric" discount factors D_t for t in 0,1,3,6,12,24.
    dfx = {
        0: discount_factors_0,
        1: discount_factors_1,
        3: discount_factors_3,
        6: discount_factors_6,
        12: discount_factors_12,
        24: discount_factors_24,
    }
    copula_spec[version]['discount_factors'] = dfx

    if df_other in ['equal_univariate']:
        # We use the parametric restrictions on c_t derived from theory.
        dict_unrestricted = None
    elif df_other in ['free']:
        # The weight c_t in the CES function is free.
        dict_unrestricted = {
            0: unrestricted_weights_0,
            1: unrestricted_weights_1,
            3: unrestricted_weights_3,
            6: unrestricted_weights_6,
            12: unrestricted_weights_12,
            24: unrestricted_weights_24,
        }
        if None in dict_unrestricted.values():
            raise TrempyError('discount function for other is set to free but contains None type')
    elif df_other in ['linear']:
        # Impose a linear structure on c_t in the CES function.
        dict_unrestricted = {
            t: max(0, y_scale + t * unrestricted_weights_0) for t in [0, 1, 3, 6, 12, 24]
        }
    elif df_other in ['exponential']:
        # Impose an exponential structure on c_t in the CES function.
        dict_unrestricted = {t: y_scale * unrestricted_weights_0 ** t for t in [0, 1, 3, 6, 12, 24]}

    # The model becomes stationary.
    if stationary_model is True:
        dict_unrestricted = {key: y_scale for key in [0, 1, 3, 6, 12, 24]}

    copula_spec[version]['unrestricted_weights'] = dict_unrestricted

    # Build copula
    copula = UtilityCopulaCls(copula_spec)

    return copula
