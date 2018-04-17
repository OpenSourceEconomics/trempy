"""This module contains auxiliary functions for the scripts."""
import argparse


def process_command_line_arguments(which):
    """This function processes the command line arguments for the test battery."""
    is_single, is_init, is_start = False, False, False

    if which == 'estimate':
        msg = 'Run estimation'
        is_single, is_start, is_init = [True] * 3
    elif which == 'simulate':
        msg = 'Run simulation'
        is_init = True
    elif which == 'update':
        msg = 'Update initialization file'
        is_init = True
    elif which == 'stop':
        msg = 'Stop estimation'
    else:
        raise NotImplementedError

    parser = argparse.ArgumentParser(msg)

    if is_single:
        parser.add_argument('--single', action='store_true', dest='is_single', required=False,
                            help='single evaluation at starting values')

    if is_init:
        parser.add_argument('--init', action='store', dest='init', type=str,
                            help='initialization file', default='model.trempy.ini')

    if is_start:
        parser.add_argument('--start', action='store', dest='start', help='starting values',
                            choices=['auto', 'init'])

    return parser.parse_args()


def distribute_command_line_arguments(args):
    """This function distributes the command line arguments."""
    rslt = dict()
    try:
        rslt['start'] = args.start
    except AttributeError:
        pass

    try:
        rslt['init'] = args.init
    except AttributeError:
        pass

    try:
        rslt['is_single'] = args.is_single
    except AttributeError:
        pass

    return rslt
