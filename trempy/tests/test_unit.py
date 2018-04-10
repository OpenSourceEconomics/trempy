"""This module contains some unit tests."""
import numpy as np

from trempy.shared.shared_auxiliary import dist_class_attributes
# from interalpy.shared.shared_auxiliary import atemporal_utility
from trempy.tests.test_auxiliary import get_random_init
# from interalpy.shared.shared_auxiliary import luce_prob
from trempy.tests.test_auxiliary import get_bounds
from trempy.tests.test_auxiliary import get_value
# from interalpy.estimate.estimate import estimate
# from interalpy.simulate.simulate import simulate
# from interalpy.config_interalpy import NUM_PARAS
from trempy.clsModel import ModelCls
from trempy.read.read import read
from trempy import simulate, estimate
#
#
# def test_1():
#     """This test simply checks that some basic functions are properly evaluated for all valid
#     specifications of the initialization files."""
#     get_random_init()
#
#     model_obj = ModelCls('test.interalpy.ini')
#     paras_obj = dist_class_attributes(model_obj, 'paras_obj')
#     r, eta, b, nu = paras_obj.get_values('econ', 'all')
#
#     for _ in range(100):
#
#         payments = np.random.lognormal(size=2)
#         u = atemporal_utility(payments, r, eta, b)
#         np.testing.assert_equal(u >= 0, True)
#
#         u_x, u_y = np.random.lognormal(size=2)
#         prob = luce_prob(u_x, u_y, nu)
#         np.testing.assert_almost_equal(np.sum(prob), 1.0)
#         np.testing.assert_equal(np.all(prob) >= 0, True)
#
#
# def test_2():
#     """This test checks the special case of linear atemporal utility."""
#     r, eta, b = 0, 0, 1
#     for _ in range(1000):
#         payment = np.random.lognormal(size=2)
#         stat = atemporal_utility(payment, r, eta, b)
#         np.testing.assert_equal(stat, payment.sum())


def test_3():
    """This test checks that the random initialization files can all be properly processed."""
    for _ in range(100):
        get_random_init()
        read('test.trempy.ini')


def test_4():
    """This test ensures the back an forth transformations for the parameter values."""
    get_random_init()

    model_obj = ModelCls('test.trempy.ini')
    paras_obj, num_questions = dist_class_attributes(model_obj, 'paras_obj', 'num_questions')

    for _ in range(500):
        x_optim_all_current = np.random.uniform(-1, 1, size=num_questions + 3)
        paras_obj.set_values('optim', 'all', x_optim_all_current)

        x_econ_all_current = paras_obj.get_values('econ', 'all')
        paras_obj.set_values('econ', 'all', x_econ_all_current)

        stat = paras_obj.get_values('optim', 'all')
        np.testing.assert_almost_equal(x_optim_all_current, stat)


def test_5():
    """This test ensures that writing out an initialization results in exactly the same value of
    the criterion function."""
    get_random_init()
    simulate('test.trempy.ini')
    x, _ = estimate('test.trempy.ini')

    model_obj = ModelCls('test.trempy.ini')
    model_obj.write_out('alt.trempy.ini')
    y, _ = estimate('alt.trempy.ini')

    np.testing.assert_almost_equal(y, x)


def test_6():
    """This test checks for valid bounds."""
    for _ in range(1000):
        for label in ['alpha', 'beta', 'eta']:
            lower, upper = get_bounds(label)
            value = get_value((lower, upper))
            np.testing.assert_equal(lower < value < upper, True)