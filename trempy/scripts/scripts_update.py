#!/usr/bin/env python
import shlex

from trempy.scripts.scripts_auxiliary import distribute_command_line_arguments
from trempy.scripts.scripts_auxiliary import process_command_line_arguments
from trempy.clsModel import ModelCls


def run(args):
    """This function updates the initialization file."""
    args = distribute_command_line_arguments(args)

    x_econ_all_step = []

    with open('est.trempy.info') as infile:

        for line in infile.readlines():

            list_ = shlex.split(line)

            # We only care about lines with the parameter values.
            if len(list_) != 5:
                continue
            try:
                x_econ_all_step += [float(list_[3])]
            except ValueError:
                pass

    model_obj = ModelCls(args['init'])
    model_obj.update('econ', 'all', x_econ_all_step)
    model_obj.write_out(args['init'])


if __name__ == '__main__':

    args = process_command_line_arguments('update')

    run(args)

