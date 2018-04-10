"""This module contains function solely related to the estimation of the model."""
import shutil
import copy
import os

from trempy import simulate
from trempy.shared.shared_auxiliary import dist_class_attributes
from trempy.config_trempy import HUGE_FLOAT


def estimate_cleanup():
    """This function ensures that we start the estimation with a clean slate."""
    # We remove the directories that contain the simulated choice menus at the start.
    for dirname in ['start', 'stop']:
        if os.path.exists(dirname):
            shutil.rmtree(dirname)

    # We remove the information from earlier estimation runs.
    for fname in ['est.trempy.info', 'est.trempy.log']:
        if os.path.exists(fname):
            os.remove(fname)


def estimate_simulate(which, points, model_obj, df_obs):
    """This function allows to easily simulate samples at the beginning and the end of the
    estimation."""
    sim_agents = dist_class_attributes(model_obj, 'sim_agents')

    os.mkdir(which)
    os.chdir(which)

    sim_model = copy.deepcopy(model_obj)
    sim_model.attr['sim_file'] = which

    sim_model.update('optim', 'free', points)
    sim_model.write_out(which + '.trempy.ini')
#    simulate(which + '.trempy.ini')

    #compare_datasets(which, df_obs, sim_agents)

    os.chdir('../')

def char_floats(floats):
    """This method ensures a pretty printing of all floats."""
    # We ensure that this function can also be called on for a single float value.
    if isinstance(floats, float):
        floats = [floats]

    line = []
    for value in floats:
        if abs(value) > HUGE_FLOAT:
            line += ['{:>25}'.format('---')]
        else:
            line += ['{:25.15f}'.format(value)]

    return line
