"""This module contains function solely related to the estimation of the model."""
import shutil
import copy
import os

from PyPDF2 import PdfFileMerger
import matplotlib.pyplot as plt
import seaborn as sns
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


def write_info_estimation(df_obs, df_finish):
    """This function writes out information about the estimation results."""
    df_finish = df_finish[abs(df_finish['Compensation']) < 1000]
    questions = sorted(df_obs['Question'].unique())

    fnames = []
    for q in questions:
        m_obs = df_obs['Compensation'][df_obs['Question'] == q]
        m_finish = df_finish['Compensation'][df_finish['Question'] == q]
        plt.subplots()
        for i, a in enumerate([m_finish, m_obs]):
            if i == 0:
                label = 'Simulated'
            elif i == 1:
                label = 'Observed'
            sns.distplot(a, label=label).set_title('Question ' + str(q))

        plt.legend()

        fname = 'model_fit_question_' + str(q) + '.pdf'
        plt.savefig(fname)

        fnames += [fname]
    plt.close('all')

    merger = PdfFileMerger()

    for fname in fnames:
        merger.append(fname)

        os.remove(fname)

    merger.write("model_fit.pdf")


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
