#!/usr/bin/env python
"""This module is the first attempt to start some regression tests."""
import pickle as pkl
import shutil

import numpy as np

from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.shared.shared_auxiliary import criterion_function
from auxiliary_tests import distribute_command_line_arguments
from trempy.tests.test_regression import run_regression_test
from trempy.tests.test_auxiliary import get_random_init
from auxiliary_tests import process_command_line_arguments
from trempy.config_trempy import PACKAGE_DIR
from auxiliary_tests import send_notification
from trempy.clsModel import ModelCls
from auxiliary_tests import cleanup
from trempy import simulate


def create_regression_vault(num_tests):
    """Create a set of regression tests."""
    np.random.seed(123)

    tests = []
    for _ in range(num_tests):

        print('\n ... creating test ' + str(_))

        constr = dict()
        constr['maxfun'] = np.random.random_integers(1, 5)

        # Create and process initialization file
        init_dict = get_random_init(constr)
        model_obj = ModelCls('test.trempy.ini')
        df, _ = simulate('test.trempy.ini')

        # Distribute class attributes for further processing.
        args = [model_obj, 'paras_obj', 'questions', 'cutoffs', 'version']
        paras_obj, questions, cutoffs, version = dist_class_attributes(*args)

        # Handle version-specific objects not included in the para_obj
        if version in ['scaled_archimedean']:
            upper, marginals = dist_class_attributes(*[model_obj, 'upper', 'marginals'])
            version_specific = {'upper': upper, 'marginals': marginals}
        elif version in ['nonstationary', 'warmglow']:
            version_specific = dict()

        # Get number of economic parameters. Paras with higher index belong to questions.
        nparas_econ = paras_obj.attr['nparas_econ']

        # Now get correct standard deviations. Versions are handled implicitly.
        x_econ_all = paras_obj.get_values('econ', 'all')
        sds = x_econ_all[nparas_econ:]

        # Evaluate criterion function and process results
        stat, _ = criterion_function(df, questions, cutoffs, paras_obj,
                                     version, sds, **version_specific)
        tests += [(init_dict, stat)]

        cleanup()

    pkl.dump(tests, open('regression_vault.trempy.pkl', 'wb'))


def check_regression_vault(num_tests):
    """Check an existing regression tests."""
    fname = PACKAGE_DIR + '/tests/regression_vault.trempy.pkl'
    tests = pkl.load(open(fname, 'rb'))

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
        if args['is_update']:
            shutil.copy('regression_vault.trempy.pkl', PACKAGE_DIR + '/tests')


if __name__ == '__main__':

    args = process_command_line_arguments('regression')

    run(args)
