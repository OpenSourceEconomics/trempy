"""This module manages everything related to interfacing the copulpy package."""
from copulpy import UtilityCopulaCls


def get_copula(upper, r_self, r_other, delta, self, other):
    """This function allows to access a multiattribute utility copula."""
    copula_spec = dict()

    copula_spec['r'] = [r_self, r_other]
    copula_spec['u'] = self, other
    copula_spec['bounds'] = upper
    copula_spec['delta'] = delta

    copula_spec['generating_function'] = 1
    copula_spec['version'] = 'scaled_archimedean'
    copula_spec['a'] = 1.0
    copula_spec['b'] = 0.0

    copula = UtilityCopulaCls(copula_spec)

    return copula
