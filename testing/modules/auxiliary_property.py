"""This module contains some auxiliary functions for the property testing."""
from datetime import datetime
import importlib
import shutil
import glob
import os

import numpy as np

from trempy.config_trempy import PACKAGE_DIR


def run_property_test(module, test, dirname=None):
    """This function runs a single robustness test."""
    mod = importlib.import_module('trempy.tests.' + module.replace('.py', ''))
    test_fun = getattr(mod, test)

    # We do not switch directories if we are investigating a failed test case.
    if dirname is not None:
        if os.path.exists(dirname):
            shutil.rmtree(dirname)

        os.mkdir(dirname)
        os.chdir(dirname)

    test_fun()


def collect_tests():
    """This function collects all available tests."""

    current_wd = os.getcwd()
    os.chdir(PACKAGE_DIR + '/tests')

    test_modules = glob.glob('test_*.py')
    test_modules.remove('test_auxiliary.py')
    os.chdir(current_wd)
    test_dict = dict()
    for module in sorted(test_modules):
        test_dict[module] = []
        mod = importlib.import_module('trempy.tests.' + module.replace('.py', ''))
        for candidate in sorted(dir(mod)):
            if 'test_' in candidate:
                test_dict[module] += [candidate]
    return test_dict


def print_rslt_ext(start, timeout, rslt, err_msg):
    """This function print out the current state of the property tests."""

    start_time = start.strftime("%Y-%m-%d %H:%M:%S")
    end_time = (start + timeout).strftime("%Y-%m-%d %H:%M:%S")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open('property.trempy.info', 'w') as outfile:

        # Write out some header information.
        outfile.write('\n\n')
        str_ = '\t{0[0]:<15}{0[1]:<20}\n\n'
        outfile.write(str_.format(['START', start_time]))
        outfile.write(str_.format(['FINISH', end_time]))
        outfile.write(str_.format(['UPDATE', current_time]))

        modules = sorted(rslt.keys())

        for module in modules:

            outfile.write('\n ' + module.replace('.py', '') + '\n\n')

            string = '{:>18}{:>15}{:>15}\n\n'
            outfile.write(string.format('Test', 'Success', 'Failure'))

            for test in sorted(rslt[module].keys()):

                stat = rslt[module][test]

                string = '{:>18}{:>15}{:>15}\n'
                outfile.write(string.format(*[test] + stat))

            outfile.write('\n')
        outfile.write('\n' + '-' * 79 + '\n\n')

        for err in err_msg:

            module, test, seed, msg = err

            string = 'MODULE {:<25} TEST {:<15} SEED {:<15}\n\n'
            outfile.write(string.format(*[module, test, seed]))
            outfile.write(msg)
            outfile.write('\n' + '-' * 79 + '\n\n')


def finish(rslt):
    """This function simply finalizes the logging."""
    # We want to record the overall performance.
    num_tests_total, num_success = 0, 0
    for module in sorted(rslt.keys()):
        for test in sorted(rslt[module].keys()):
            num_tests_total += np.sum(rslt[module][test])
            num_success += rslt[module][test][0]

    with open('property.trempy.info', 'a') as outfile:
        string = '{:>18}{:>15}\n'
        outfile.write(string.format(*['Success', num_tests_total]))
        outfile.write(string.format(*['Total', num_success]))

        outfile.write('\n TERMINATED')
