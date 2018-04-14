#!/usr/bin/env python
"""This script allows to properly shut down the estimation process."""
from trempy.scripts.scripts_auxiliary import process_command_line_arguments


def run():
    """This function stps the estimation"""
    open('.stop.trempy.scratch', 'a').close()


if __name__ == '__main__':

    process_command_line_arguments("stop")

    run()