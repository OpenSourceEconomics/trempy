#!/usr/bin/env python
import numpy as np

from trempy.clsModel import ModelCls
from trempy import estimate
from trempy import simulate
from trempy.tests.test_auxiliary import get_random_init, get_rmse

model_obj = ModelCls('model.trempy.ini')
simulate('model.trempy.ini')
estimate('model.trempy.ini')
