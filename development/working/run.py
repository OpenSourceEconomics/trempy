#!/usr/bin/env python
from trempy.clsModel import ModelCls
from trempy import estimate
from trempy import simulate

model_obj = ModelCls('model.trempy.ini')
simulate('model.trempy.ini')
estimate('model.trempy.ini')
