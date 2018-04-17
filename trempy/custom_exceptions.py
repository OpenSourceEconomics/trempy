"""This module contains the custom exceptions for the package."""


class MaxfunError(Exception):
    """This custom exception is raised if the maximum number of function evaluations is reached."""
    def __init__(self):
        pass


class TrempyError(Exception):
    """This custom exception is used throughout the package."""
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return '\n\n ... {}\n'.format(self.message)
