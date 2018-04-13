"""This module contains all functions related to the tests on the observed dataset."""
import functools

import numpy as np

from trempy.config_trempy import NEVER_SWITCHERS


def process_checks(df, est_agents, questions, cutoffs):
    """This function performs numerous checks on the observed dataset."""
    # We want the index properly set up to individual and questions.
    np.testing.assert_equal(df.index.names, ['Individual', 'Question'])

    # We need the information on the level of compensation.
    np.testing.assert_equal('Compensation' in df.columns, True)

    # We need enough individuals to run the estimation on the number of individuals requested.
    num_obs = df.index.get_level_values(0).nunique()
    np.testing.assert_equal(num_obs >= est_agents, True)

    # We want all individuals for all questions.
    def _check_questions(questions, agent):
        """This function checks whether all questions are defined for all individuals."""
        idx = sorted(agent.index.get_level_values(1))
        np.testing.assert_equal(idx, questions)
    df.groupby('Individual').apply(functools.partial(_check_questions, questions))

    # Check that compensation levels line up with cutoffs and the NEVER_SWITCHERS
    for q in questions:
        lower, upper = cutoffs[q]
        cond = df.loc[(slice(None), q), 'Compensation'].between(lower, upper, inclusive=True)
        cond = cond | (df.loc[(slice(None), q), 'Compensation'].isin([NEVER_SWITCHERS]))
        np.testing.assert_equal(np.all(cond), True)
