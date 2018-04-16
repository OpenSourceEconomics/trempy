import importlib

from auxiliary_property import collect_tests

test_dict = collect_tests()


# Now I can run a random test.
seed = 73140
module = 'test_integration.py'
test = 'test_2'

mod = importlib.import_module('trempy.tests.' + module.replace('.py', ''))
test_fun = getattr(mod, test)

test_fun()
