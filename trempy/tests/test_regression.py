"""This module contains some integration tests."""
import pickle as pkl

import numpy as np

from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.shared.shared_auxiliary import criterion_function
from trempy.shared.shared_auxiliary import print_init_dict
from trempy.config_trempy import TEST_RESOURCES_DIR
from trempy.clsModel import ModelCls
from trempy import simulate


def run_regression_test(test):
    """This function runs a single regression test. It is repeatedly used by the testing
    infrastructure. Thus, manual modifications are only required here."""
    # Create and process initialization file
    init_dict, crit_val = test

    print_init_dict(init_dict)
    model_obj = ModelCls('test.trempy.ini')
    df = simulate('test.trempy.ini')

    # Distribute class attributes for further processing.
    paras_obj, questions, cutoffs = dist_class_attributes(model_obj, 'paras_obj', "questions",
        'cutoffs')

    x_econ_all = paras_obj.get_values('econ', 'all')
    stat = criterion_function(df, questions, cutoffs, *x_econ_all)

    np.testing.assert_almost_equal(stat, crit_val)


def test_1():
    """This test simply runs a small sample of the regression test battery."""
    tests = pkl.load(open(TEST_RESOURCES_DIR + '/regression_vault.trempy.pkl', 'rb') )

    for test in tests[:5]:
        run_regression_test(test)
