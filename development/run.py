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


simulate('simulate.trempy.ini')
estimate('simulate.trempy.ini')

import cProfile
cProfile.run("estimate('simulate.trempy.ini')", 'profile.prof')
