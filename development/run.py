#!/usr/bin/env python
import filecmp

import numpy as np
import random

from trempy.clsModel import ModelCls
from trempy import estimate
from trempy import simulate
from trempy.tests.test_auxiliary import get_random_init, get_rmse
from trempy.read.read import read
import os
from trempy.paras.clsParas import ParasCls


estimate('model.trempy.ini')
#
# for _ in range(1):
#     np.random.seed(1423)
#     constr = dict()
#     constr['maxfun'] = 0 # np.random.random_integers(1, 5)
#
#     get_random_init(constr)
#     print("simulate")
#     simulate('test.trempy.ini')
#     print("estimate")
#     f_step, _ = estimate('test.trempy.ini')
#
#     np.testing.assert_equal(f_step, 1.6327683449552237)
