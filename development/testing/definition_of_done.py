#!/usr/bin/env python
"""This script implements parts of our definition of done."""
from functools import partial
import multiprocessing as mp
import subprocess
import sys
import os

num_tests = 1000
minutes = 60


def run(minutes, num_tests, dirname):
    """This functions runs each of the tests groups."""
    os.chdir(dirname)

    # This is required as otherwise I am not using the virtual environment on the acropolis cluster.
    PYTHON_EXEC = sys.executable

    if dirname == 'property':
        cmd, arg = PYTHON_EXEC + ' run.py --minutes {:f}', minutes
    elif dirname == 'regression':
        cmd, arg = PYTHON_EXEC + ' run.py --request check --tests {:d}', num_tests
    else:
        raise NotImplementedError

    subprocess.check_call(cmd.format(arg).split(' '))
    os.chdir('../')


results = mp.Pool(2).map(partial(run, minutes, num_tests), ['regression', 'property'])
