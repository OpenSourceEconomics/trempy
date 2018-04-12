#!/usr/bin/env python
"""This script allows to run an simulation from the command line."""
import argparse

from trempy.simulate.simulate import simulate

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Run simulation')

    parser.add_argument('--init', action='store', dest='fname', type=str,
        help='initialization file', default='model.trempy.ini')


    args = parser.parse_args()

    simulate(args.fname)
