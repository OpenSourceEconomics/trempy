"""Manage everything related to interfacing the copulpy package."""
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
                             unrestricted_weights_12, unrestricted_weights_24):
    """Access the nonstationary utility copula."""
    version = 'nonstationary'
    copula_spec = {'version': version}
    copula_spec[version] = {
        'version': version,
        'y_scale': y_scale,
        'alpha': alpha,
        'gamma': gamma,
        'beta': beta,
    }

    # Discount factors
    dict_discount_f = {
        0: discount_factors_0,
        1: discount_factors_1,
        3: discount_factors_3,
        6: discount_factors_6,
        12: discount_factors_12,
        24: discount_factors_24,
    }
    copula_spec[version]['discount_factors'] = dict_discount_f

    # Optional argument: unrestricted weights
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
    copula_spec[version]['unrestricted_weights'] = dict_unrestriced

    # Build copula
    copula = UtilityCopulaCls(copula_spec)

    return copula
