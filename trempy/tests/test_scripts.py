"""This module contains some scripts tests."""
from trempy.scripts.scripts_estimate import run as run_estimate
from trempy.scripts.scripts_simulate import run as run_simulate
from trempy.scripts.scripts_update import run as run_update
from trempy.tests.test_auxiliary import get_random_init


class Object(object):
    pass


def test_1():
    """This test just runs the scripts."""
    # This object mimics the argument parser object.
    args = Object()
    args.start = get_random_init()['ESTIMATION']['start']
    args.is_single = True
    args.init = 'test.trempy.ini'

    run_simulate(args)
    run_estimate(args)
    run_update(args)
