#!/usr/bin/env python
"""This module executes a complete testing of the package."""
import subprocess
import sys
import os

# Specification
request = dict()

request['property'] = dict()
request['property']['run'] = True
request['property']['hours'] = 12

request['regression'] = dict()
request['regression']['run'] = True
request['regression']['tests'] = 1000

####################################################################################################
####################################################################################################

PYTHON_EXEC = sys.executable

# property-based testing
if request['property']['run']:
    os.chdir('property')
    cmd = PYTHON_EXEC + ' run.py --request run --hours ' + str(request['property']['hours'])
    subprocess.check_call(cmd, shell=True)
    os.chdir('../')

# regression testing
if request['regression']['run']:
    os.chdir('regression')
    cmd = PYTHON_EXEC + ' run.py --request check --tests ' + str(request['regression']['tests'])
    subprocess.check_call(cmd, shell=True)
    os.chdir('../')
