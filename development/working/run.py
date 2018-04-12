#!/usr/bin/env python
import filecmp

import numpy as np

from trempy.clsModel import ModelCls
from trempy import estimate
from trempy import simulate
from trempy.tests.test_auxiliary import get_random_init, get_rmse

np.random.seed(1232)
count = 0
while True:
    print(' Count ', count)
    count += 1
    init_dict = get_random_init()

    simulate('test.trempy.ini')
    x, _ = estimate('test.trempy.ini')

    model_obj = ModelCls('test.trempy.ini')
    model_obj.write_out('alt.trempy.ini')
    y, _ = estimate('alt.trempy.ini')

    np.testing.assert_equal(filecmp.cmp('test.trempy.ini', 'alt.trempy.ini'), True)
    np.testing.assert_almost_equal(y, x)
