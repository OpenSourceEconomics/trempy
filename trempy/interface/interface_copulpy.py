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


def get_copula_nonstationary(alpha, beta, gamma, y_scale, discont_factors,
                             restricted=True, unrestricted_weights=None):
    """Access the nonstationary utility copula."""
    copula_spec = dict()
    copula_spec['alpha'] = alpha
    copula_spec['beta'] = beta
    copula_spec['gamma'] = gamma
    copula_spec['y_scale'] = y_scale
    copula_spec['discont_factors'] = discont_factors
    copula_spec['restricted'] = restricted
    copula_spec['unrestricted_weights'] = unrestricted_weights

    copula_spec['version'] = 'nonstationary'

    copula = UtilityCopulaCls(copula_spec)

    return copula
