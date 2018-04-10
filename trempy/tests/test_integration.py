"""This module contains some integration tests."""
# import numpy as np
#
# from interalpy.shared.shared_auxiliary import dist_class_attributes
# from interalpy.tests.test_auxiliary import get_random_init
# from interalpy.shared.shared_auxiliary import solve_grid
# from interalpy.tests.test_auxiliary import get_rmse
# from interalpy import simulate
# from interalpy import ModelCls
# from interalpy import estimate
#
#
# def test_1():
#     """This test simply runs the core workflow of simulation and estimation."""
#     for _ in range(5):
#         get_random_init()
#         simulate('test.interalpy.ini')
#         estimate('test.interalpy.ini')
#
#
# def test_2():
#     """This test checks the integrity of the solution grid."""
#     for _ in range(5):
#         get_random_init()
#         model_obj = ModelCls('test.interalpy.ini')
#
#         # Distribute class attributes for further processing.
#         paras_obj = dist_class_attributes(model_obj, 'paras_obj')
#         r, eta, b, nu = paras_obj.get_values('econ', 'all')
#         df = solve_grid(r, eta, b, nu)
#
#         # The probabilities will need to sum to one.
#         stat = df[['prob_a', 'prob_b']].sum(axis=1)
#         np.testing.assert_equal(np.all(stat) == 1.0, True)
#
#         # There is variation in in the expected utility for lottery A.
#         stat = df['eu_a'].groupby(level='Question').nunique()
#         np.testing.assert_equal(np.all(stat) == 1.0, True)
#
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