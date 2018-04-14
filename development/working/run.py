#!/usr/bin/env python
import filecmp

import numpy as np

from trempy.clsModel import ModelCls
from trempy import estimate
from trempy import simulate
from trempy.tests.test_auxiliary import get_random_init, get_rmse
import os

np.random.seed(2)
for _ in range(1):
    os.system('git clean -d -f')
    get_random_init()
    #simulate('test.trempy.ini')
    #estimate('test.trempy.ini')
