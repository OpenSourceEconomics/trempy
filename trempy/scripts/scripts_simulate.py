#!/usr/bin/env python
"""This script allows to run an simulation from the command line."""
from trempy.scripts.scripts_auxiliary import distribute_command_line_arguments
from trempy.scripts.scripts_auxiliary import process_command_line_arguments
from trempy.simulate.simulate import simulate


def run():
    """This function allows to start a simulation."""
    args = distribute_command_line_arguments(args)

    simulate(args['init'])


if __name__ == '__main__':

    args = process_command_line_arguments('simulate')

    run(args)
