"""This module contains all function related to the processing of the observed dataset."""
import pandas as pd

from trempy.process.process_checks import process_checks


def process(est_file, questions, num_skip, est_agents, cutoffs):
    """This function processes the observed dataset."""
    df = pd.read_pickle(est_file)

    # We cut the dataset to only contain the information that is actually used.
    df = df.loc[(slice(None), slice(None)), 'Compensation'].to_frame()
    lower, upper = int(num_skip), int(num_skip + est_agents)
    subset = df.index.get_level_values(0).unique()[lower:upper]
    df = df.loc[(subset, questions), :]

    # We perform some tests on the dataset.
    process_checks(df, est_agents, questions, cutoffs)

    return df
