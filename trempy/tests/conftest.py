"""This module provides the fixtures for the PYTEST runs."""
import numpy as np

import tempfile
import pytest
import os


@pytest.fixture(scope="module", autouse=True)
def set_seed():
    """Each test is executed with the same random seed."""
    np.random.seed(1423)


@pytest.fixture(scope="module", autouse=True)
def fresh_directory():
    """Each test is executed in a fresh directory."""
    os.chdir(tempfile.mkdtemp())
