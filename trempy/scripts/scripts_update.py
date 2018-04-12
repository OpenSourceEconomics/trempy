#!/usr/bin/env python
import argparse
import shlex

from trempy.clsModel import ModelCls

if __name__ == '__main__':


    parser = argparse.ArgumentParser('Update initialization file')

    parser.add_argument('--init', action='store', dest='fname', type=str,
        help='initialization file', default='model.trempy.ini')


    args = parser.parse_args()
    fname = args.fname

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

    model_obj = ModelCls(fname)
    model_obj.update('econ', 'all', x_econ_all_step)
    model_obj.write_out(fname)

