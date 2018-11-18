"""Manage everything related to interfacing the copulpy package."""
from copulpy import UtilityCopulaCls


def get_copula(upper, marginals, r_self, r_other, delta, self, other):
    """Access a multiattribute utility copula."""
    copula_spec = dict()

    copula_spec['r'] = [r_self, r_other]
    copula_spec['marginals'] = marginals
    copula_spec['u'] = self, other
    copula_spec['bounds'] = upper
    copula_spec['delta'] = delta

    copula_spec['generating_function'] = 1
    copula_spec['version'] = 'scaled_archimedean'
    copula_spec['a'] = 1.0
    copula_spec['b'] = 0.0

    copula = UtilityCopulaCls(copula_spec)

    return copula


def get_copula_nonstationary(alpha, beta, gamma, y_scale,
                             discount_factors_0, discount_factors_1, discount_factors_3,
                             discount_factors_6, discount_factors_12, discount_factors_24,
                             unrestricted_weights_0, unrestricted_weights_1,
                             unrestricted_weights_3, unrestricted_weights_6,
                             unrestricted_weights_12, unrestricted_weights_24):
    """Access the nonstationary utility copula."""
    copula_spec = dict()
    copula_spec['version'] = 'nonstationary'
    copula_spec['alpha'] = alpha
    copula_spec['beta'] = beta
    copula_spec['gamma'] = gamma
    copula_spec['y_scale'] = y_scale

    # Discount factors
    dict_discount_f = {
        0: discount_factors_0,
        1: discount_factors_1,
        3: discount_factors_3,
        6: discount_factors_6,
        12: discount_factors_12,
        24: discount_factors_24,
    }

    dict_unrestriced = {
        0: unrestricted_weights_0,
        1: unrestricted_weights_1,
        3: unrestricted_weights_3,
        6: unrestricted_weights_6,
        12: unrestricted_weights_12,
        24: unrestricted_weights_24,
    }

    if None in dict_unrestriced.values():
        dict_unrestriced = None

    # Optional argument: unrestricted weights
    copula_spec['discount_factors'] = dict_discount_f
    copula_spec['unrestricted_weights'] = dict_unrestriced

    # Build copula
    copula = UtilityCopulaCls(copula_spec)

    return copula
