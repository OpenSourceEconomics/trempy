"""This module contains some integration tests."""
import numpy as np

#from interalpy.shared.shared_auxiliary import dist_class_attributes
from trempy.tests.test_auxiliary import get_random_init
#from interalpy.shared.shared_auxiliary import solve_grid
#from interalpy.tests.test_auxiliary import get_rmse
from trempy import simulate
from trempy.clsModel import ModelCls
from trempy import estimate


def test_1():
    """This test simply runs the core workflow of simulation and estimation."""
    for _ in range(5):
        get_random_init()
        simulate('test.trempy.ini')
#        estimate('test.trempy.ini')
#
# def test_3():
#     """This test ensures that using the same initialization file for a simulation and a single
#     evaluation of the criterion function result in the very same simulated sample at the stop of
#     the estimation"""
#     constr = dict()
#     constr['num_agents'] = np.random.randint(2, 10)
#     constr['detailed'] = 'True'
#     constr['maxfun'] = 1
#
#     for _ in range(5):
#         get_random_init(constr)
#         simulate('test.interalpy.ini')
#         estimate('test.interalpy.ini')
#         np.testing.assert_equal(get_rmse(), 0.0)