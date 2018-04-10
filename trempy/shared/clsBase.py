"""This module contains the baseline class."""


class BaseCls(object):
    """This class provides some basic capabilities for all the project's classes"""
    def __init__(self):
        pass

    def get_attr(self, key):
        """This method allows to access class attribute."""
        return self.attr[key]

    def set_attr(self, key, value):
        """This method allows to set the value of a class attribute."""
        self.attr[key] = value
