"""This module manages everything related to interfacing the copualpy package."""
from copulpy import UtilityCopulaCls


def get_copula(r_self, r_other, delta, self, other):
    """This function allows to access a multiattribute utility copula."""

    copula_spec = dict()

    copula_spec['r'] = [r_self, r_other]
    copula_spec['delta'] = delta
    copula_spec['u'] = self, other

    copula_spec['generating_function'] = 1
    copula_spec['version'] = 'scaled_archimedean'

    # TODO: These need to be integarated
    # TODO: Logging of successful optimization in copula fit
    copula_spec['bounds'] = 200, 200
    copula_spec['a'] = 1.0
    copula_spec['b'] = 0.0

    copula = UtilityCopulaCls(copula_spec)


    return copula