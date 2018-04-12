#!/usr/bin/env python
import filecmp

import numpy as np

from trempy.clsModel import ModelCls
from trempy import estimate
from trempy import simulate
from trempy.tests.test_auxiliary import get_random_init, get_rmse

np.random.seed(1423)
init_dict = get_random_init()
simulate('test.trempy.ini')
estimate('test.trempy.ini')
