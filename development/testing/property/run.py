#!/usr/bin/env python
"""This module is a first stab at property testing."""
import argparse
import random
import time
import sys
sys.path.insert(0, '../modules')

import numpy as np

from testing_shared import create_random_init
from testing_shared import run_test_case
from testing_shared import cleanup

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Run property testing')

    parser.add_argument('--minutes', action='store', dest='num_minutes', type=float, required=True,
                        help='number of minutes to run')

    args = parser.parse_args()

    timeout_start = time.time()

    while time.time() < timeout_start + args.num_minutes * 60:

        seed = random.randint(0, 10000)
        np.random.seed(seed)

        try:
            init_dict = create_random_init()
            run_test_case(init_dict)
        except:
            raise AssertionError(' ... failure with seed ', seed)

    cleanup()
