#!/usr/bin/env python
"""This script allows to run an estimation from the command line."""
import os

from trempy.scripts.scripts_auxiliary import distribute_command_line_arguments
from trempy.scripts.scripts_auxiliary import process_command_line_arguments
from trempy.estimate.estimate import estimate
from trempy.clsModel import ModelCls


def run(args):
    """This function allows to start an estimation."""
    args = distribute_command_line_arguments(args)

    model_obj = ModelCls(args['init'])

    if args['is_single']:
        model_obj.set_attr('maxfun', 1)

    if args['start'] is not None:
        model_obj.set_attr('start', args['start'])

    model_obj.write_out('.tmp.trempy.ini')
    estimate('.tmp.trempy.ini')
    os.remove('.tmp.trempy.ini')


if __name__ == '__main__':

    args = process_command_line_arguments('estimate')

    run(args)

