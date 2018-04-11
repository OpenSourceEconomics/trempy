"""This module contains some auxiliary functions that are used throughout the test battery."""
import subprocess
import argparse
import socket
import os

from clsMail import MailCls


def distribute_command_line_arguments(args):
    """This function distributes the command line arguments."""
    rslt = dict()
    try:
        rslt['num_tests'] = args.num_tests
    except AttributeError:
        pass

    try:
        rslt['request'] = args.request
    except AttributeError:
        pass

    try:
        rslt['hours'] = args.hours
    except AttributeError:
        pass

    try:
        rslt['seed'] = args.seed
    except AttributeError:
        pass

    rslt['is_check'] = rslt['request'] in ['check', 'investigate']

    return rslt


def cleanup():
    """This removes all nuisance files."""
    subprocess.check_call(['git', 'clean', '-d', '-f', '-q'])


def process_command_line_arguments(which):
    """This function processes the command line arguments for the test battery."""
    is_request, is_hours, is_seed, is_test = False, False, False, False

    if which == 'robustness':
        msg = 'Test robustness of package'
        is_request, is_hours, is_seed = True, True, True
    elif which == 'regression':
        msg = 'Test package for regressions'
        is_request, is_test = True, True
    elif which == 'property':
        msg = 'Property testing of package'
        is_request , is_seed, is_hours = True, True, True
    else:
        raise NotImplementedError

    parser = argparse.ArgumentParser(msg)

    if is_request:
        if which == 'regression':
            parser.add_argument('--request', action='store', dest='request', help='task to perform',
                                required=True, choices=['check', 'create'])
        else:
            parser.add_argument('--request', action='store', dest='request', help='task to perform',
                                required=True, choices=['run', 'investigate'])

    if is_hours:
        parser.add_argument('--hours', action='store', dest='hours', type=float, help='hours')

    if is_seed:
        parser.add_argument('--seed', action='store', dest='seed', type=int, help='seed')

    if is_test:
        parser.add_argument('--tests', action='store', dest='num_tests', required=True, type=int,
                            help='number of tests')

    return parser.parse_args()


def send_notification(which, **kwargs):
    """Finishing up a run of the testing battery."""
    # This allows to run the scripts even when no notification can be send.
    if not os.path.exists(os.environ['HOME'] + '/.credentials'):
        return

    hours, is_failed, num_tests, seed = None, None, None, None

    if 'num_tests' in kwargs.keys():
        num_tests = '{}'.format(kwargs['num_tests'])

    if 'is_failed' in kwargs.keys():
        is_failed = kwargs['is_failed']

    if 'hours' in kwargs.keys():
        hours = '{}'.format(kwargs['hours'])

    if 'seed' in kwargs.keys():
        seed = '{}'.format(kwargs['seed'])

    if 'count' in kwargs.keys():
        count = '{}'.format(kwargs['count'])

    hostname = socket.gethostname()

    if which == 'property':
        subject = ' INTERALPY: Property Testing'
        message = ' A ' + hours + ' hour run of the testing battery on @' + hostname + \
                  ' is completed.'

    elif which == 'robustness':
        subject = ' INTERALPY: Robustness Testing'
        if not is_failed:
            message = ' A ' + hours + ' hour run of the testing battery on @' + hostname + \
                      ' is completed. In total we ran ' + num_tests + ' tests.'

        else:
            message = ' Failure during robustness testing on @' + hostname + ' for test ' + \
                      seed + ' failed.'
    elif which == 'regression':
        subject = ' INTERALPY: Regression Testing'
        if is_failed:
            message = 'Failure during regression testing on @' + hostname + ' for test ' + \
                      count + '.'

        else:
            message = ' Regression testing is completed on on @' + hostname + '. In total ' + \
                      'we ran ' + num_tests + ' tests.'
    else:
        raise AssertionError

    mail_obj = MailCls()
    mail_obj.set_attr('subject', subject)
    mail_obj.set_attr('message', message)

    if which == 'property':
        mail_obj.set_attr('attachment', 'property.interalpy.info')
    mail_obj.lock()
    mail_obj.send()