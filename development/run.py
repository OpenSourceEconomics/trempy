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

np.random.seed(13)
for _ in range(10):
    get_random_init()

    simulate('test.trempy.ini')
    estimate('test.trempy.ini')

