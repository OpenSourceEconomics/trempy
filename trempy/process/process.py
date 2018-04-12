"""This module contains all function related to the processing of the observed dataset."""
import pandas as pd

from trempy.process.process_checks import process_checks


def process(est_file, questions, est_agents):
    """This function processes the observed dataset."""
    df = pd.read_pickle(est_file)

    # We perform some tests on the dataset.
    process_checks(df, est_agents)

    # We cut the dataset to only contain the information that is actually used.
    df = df.loc[(slice(None), slice(None)), 'Compensation'].to_frame()
    subset = df.index.get_level_values(0).unique()[:est_agents]
    df = df.loc[(subset, questions), :]

    return df
