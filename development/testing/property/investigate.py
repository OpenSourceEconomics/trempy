import importlib

from auxiliary_property import collect_tests

test_dict = collect_tests()


# Now I can run a random test.
seed = 1423
module = 'test_integration.py'
test = 'test_1'

mod = importlib.import_module('interalpy.tests.' + module.replace('.py', ''))
test_fun = getattr(mod, test)

test_fun()
