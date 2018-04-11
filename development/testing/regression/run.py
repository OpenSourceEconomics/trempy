#!/usr/bin/env python
"""This module is the first attempt to start some regression tests."""
import json

import numpy as np

from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.shared.shared_auxiliary import criterion_function
from auxiliary_tests import distribute_command_line_arguments
from trempy.tests.test_regression import run_regression_test
from trempy.tests.test_auxiliary import get_random_init
from auxiliary_tests import process_command_line_arguments
from trempy.config_trempy import TEST_RESOURCES_DIR
from auxiliary_tests import send_notification
from trempy.clsModel import ModelCls
from auxiliary_tests import cleanup
from trempy import simulate


def create_regression_vault(num_tests):
    """This function creates a set of regression tests."""
    np.random.seed(123)

    tests = []
    for _ in range(num_tests):

        print('\n ... creating test ' + str(_))

        # Create and process initialization file
        init_dict = get_random_init()
        model_obj = ModelCls('test.trempy.ini')
        df = simulate('test.trempy.ini')

        # We want to ensure that the keys to the questions are strings. Otherwise, serialization
        # fails.
        for label in ['QUESTIONS', 'CUTOFFS']:
            init_dict[label] = {str(x): init_dict[label][x] for x in init_dict[label].keys()}

        # Distribute class attributes for further processing.
        paras_obj, questions, cutoffs = dist_class_attributes(model_obj, 'paras_obj', 'questions',
            'cutoffs')

        x_econ_all = paras_obj.get_values('econ', 'all')
        stat = criterion_function(df, questions, cutoffs, *x_econ_all)

        tests += [(init_dict, stat)]

        cleanup()

    json.dump(tests, open('regression_vault.trempy.json', 'w'))


def check_regression_vault(num_tests):
    """This function checks an existing regression tests."""
    fname = TEST_RESOURCES_DIR + '/regression_vault.trempy.json'
    tests = json.load(open(fname, 'r'))

    for i, test in enumerate(tests[:num_tests]):
        try:
            run_regression_test(test)
        except Exception:
            send_notification('regression', is_failed=True, count=i)
            raise SystemError

        cleanup()

    send_notification('regression', is_failed=False, num_tests=num_tests)


def run(args):
    """Create or check the regression tests."""
    args = distribute_command_line_arguments(args)
    if args['is_check']:
        check_regression_vault(args['num_tests'])
    else:
        create_regression_vault(args['num_tests'])


if __name__ == '__main__':

    args = process_command_line_arguments('regression')

    run(args)
