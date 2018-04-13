#!/usr/bin/env python
"""This script allows to run an estimation from the command line."""
import argparse
import os

from trempy.estimate.estimate import estimate
from trempy.clsModel import ModelCls

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Run estimation')

    parser.add_argument('--single', action='store_true', dest='is_single', required=False,
        help='single evaluation at starting values')

    parser.add_argument('--init', action='store', dest='fname', type=str,
        help='initialization file', default='model.trempy.ini')

    args = parser.parse_args()

    base_init = args.fname

    if args.is_single:
        model_obj = ModelCls(base_init)
        model_obj.set_attr('maxfun', 1)
        model_obj.write_out('.tmp.trempy.ini')
        estimate('.tmp.trempy.ini')
        os.remove('.tmp.trempy.ini')
    else:
        estimate(base_init)
