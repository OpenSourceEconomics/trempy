"""This module contains some unit tests."""
import filecmp

import numpy as np

from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.tests.test_auxiliary import get_random_init
from trempy.config_trempy import PREFERENCE_PARAMETERS
from trempy.tests.test_auxiliary import get_bounds
from trempy.tests.test_auxiliary import get_value
from trempy.clsModel import ModelCls
from trempy.read.read import read
from trempy import simulate
from trempy import estimate


def test_1():
    """Check that the random initialization files can all be properly processed."""
    for _ in range(100):
        constr = {'fname': 'test.trempy.dict'}
        get_random_init(constr)
        read('test.trempy.ini')


def test_2():
    """Ensure the back an forth transformations for the parameter values."""
    get_random_init()

    model_obj = ModelCls('test.trempy.ini')
    paras_obj, num_questions = dist_class_attributes(model_obj, 'paras_obj', 'num_questions')
    nparas_econ = paras_obj.attr['nparas_econ']

    for _ in range(500):
        x_optim_all_current = np.random.uniform(-1, 1, size=num_questions + nparas_econ)
        paras_obj.set_values('optim', 'all', x_optim_all_current)

        x_econ_all_current = paras_obj.get_values('econ', 'all')
        paras_obj.set_values('econ', 'all', x_econ_all_current)

        stat = paras_obj.get_values('optim', 'all')
        np.testing.assert_almost_equal(x_optim_all_current, stat)


def test_3():
    """Ensure that writing out an init_dict results in the same value of the criterion function."""
    constr = {'maxfun': 2}

    get_random_init(constr)
    simulate('test.trempy.ini')
    x, _ = estimate('test.trempy.ini')

    model_obj = ModelCls('test.trempy.ini')
    model_obj.write_out('alt.trempy.ini')
    y, _ = estimate('alt.trempy.ini')

    np.testing.assert_almost_equal(y, x)


def test_4():
    """Check for valid bounds."""
    for _ in range(1000):
        version = np.random.choice(['scaled_archimedean', 'nonstationary'])

        for label in PREFERENCE_PARAMETERS[version]:
            lower, upper = get_bounds(label, version)
            value = get_value((lower, upper), label, version)
            if value is not None:
                np.testing.assert_equal(lower < value < upper, True)


def test_5():
    """Ensure that the original and printed version of the initialization file are identical."""
    get_random_init()
    model_obj = ModelCls('test.trempy.ini')
    model_obj.write_out('alt.trempy.ini')

    np.testing.assert_equal(filecmp.cmp('test.trempy.ini', 'alt.trempy.ini'), True)
