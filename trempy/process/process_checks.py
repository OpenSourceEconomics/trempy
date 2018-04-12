"""This module contains all functions related to the tests on the observed dataset."""
import numpy as np


def process_checks(df, est_agents):
    """This function performs numerous checks on the observed dataset."""
    # We want the index properly set up to individual and questions.
    np.testing.assert_equal(df.index.names, ['Individual', 'Question'])

    # We need the information on the level of compensation.
    np.testing.assert_equal('Compensation' in df.columns, True)

    # We need enough individuals to run the estimation on the number of individuals requested.
    num_obs = df.index.get_level_values(0).nunique()
    np.testing.assert_equal(num_obs >= est_agents, True)

    # TODO:
    # We want all individuals for all questions.
    # Check that compensation levels line up with cutoffs and the NEVER_SWITSCHERS
