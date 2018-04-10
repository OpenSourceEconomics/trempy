#!/usr/bin/env python
"""This script is the first take at a regression test setup."""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, '../modules')

from testing_shared import create_random_init
from testing_shared import run_test_case
from testing_shared import cleanup


def create(num_tests):
    """This function creates a new regression vault."""
    np.random.seed(123)

    tests = []
    for _ in range(num_tests):

        # Initialize test case
        init_dict = create_random_init()

        # Run test case
        fval = run_test_case(init_dict)

        # Record test case
        tests += [[init_dict, fval]]

    with open('created_vault.interalpy.json', 'w') as outfile:
        json.dump(tests, outfile)


def check(num_tests):
    """This function checks the regression vault."""
    with open('regression_vault.interalpy.json') as infile:
        tests = json.load(infile)

    for i in range(num_tests):

        # Distribute test case
        init_dict, stat = tests[i]

        # For some reason the serialization results in string keys for the cutoff values.
        questions = list(init_dict['cutoffs'].keys())
        for key_ in questions:
            init_dict['cutoffs'][int(key_)] = init_dict['cutoffs'][key_]

        # Run test case
        fval = run_test_case(init_dict)

        # Check test case
        np.testing.assert_equal(fval, stat)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Run regression testing')

    parser.add_argument('--request', action='store', dest='request', help='task to perform',
                        required=True, choices=['check', 'create'])

    parser.add_argument('--tests', action='store', dest='num_tests', type=int, required=True,
                        help='number of tests')

    args = parser.parse_args()

    if args.request == 'create':
        create(args.num_tests)
        cleanup(is_create=True)
    elif args.request == 'check':
        check(args.num_tests)
        cleanup()

