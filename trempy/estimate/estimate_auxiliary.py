"""This module contains function solely related to the estimation of the model."""
import os

from PyPDF2 import PdfFileMerger
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)


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