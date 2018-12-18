"""This module contains all function related to the processing of the observed dataset."""
import pandas as pd
import numpy as np
import functools

from trempy.config_trempy import NEVER_SWITCHERS


def process(est_file, questions, num_skip, est_agents, cutoffs):
    """Process the observed dataset."""
    df = pd.read_pickle(est_file)

    # We cut the dataset to only contain the information that is actually used.
    df = df.loc[(slice(None), slice(None)), 'Compensation'].to_frame()
    lower, upper = int(num_skip), int(num_skip + est_agents)
    subset = df.index.get_level_values(0).unique()[lower:upper]
    df = df.loc[(subset, questions), :]

    # We perform some tests on the dataset.
    process_checks(df, est_agents, questions, cutoffs)

    return df


def process_checks(df, est_agents, questions, cutoffs):
    """Perform numerous checks on the observed dataset."""
    # We want the index properly set up to individual and questions.
    np.testing.assert_equal(df.index.names, ['Individual', 'Question'])

    # We need the information on the level of compensation.
    np.testing.assert_equal('Compensation' in df.columns, True)

    # We need enough individuals to run the estimation on the number of individuals requested.
    num_obs = df.index.get_level_values(0).nunique()
    np.testing.assert_equal(num_obs >= est_agents, True)

    # We want all individuals for all questions.
    def _check_questions(questions, agent):
        """Check whether all questions are defined for all individuals."""
        idx = sorted(agent.index.get_level_values(1))
        np.testing.assert_equal(idx, questions)
    df.groupby('Individual').apply(functools.partial(_check_questions, questions))

    # Check that compensation levels line up with cutoffs and the NEVER_SWITCHERS
    for q in questions:
        lower, upper = cutoffs[q]
        print('Q: {0}, lower: {1}, upper: {2}'.format(q, lower, upper))
        print(df.loc[(slice(None), q), 'Compensation'].describe())
        print(df.loc[(slice(None), q), 'Compensation'].isin([NEVER_SWITCHERS]).sum())
        cond = df.loc[(slice(None), q), 'Compensation'].between(lower, upper, inclusive=True)

        cond = cond | (df.loc[(slice(None), q), 'Compensation'].isin([NEVER_SWITCHERS]))
        np.testing.assert_equal(np.all(cond), True)
