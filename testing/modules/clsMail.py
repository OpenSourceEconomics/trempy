"""This module holds the class for mailing capabilities.

   A file with the log in credentials is required in the HOME directory.

    Example Usage:

        mailObj = mailCls()

        mailObj.setAttr('subject', ' Example ')

        mailObj.setAttr('message', ' This is an example message. ')

        mailObj.setAttr('attachment', 'logging.test.txt')

        mailObj.lock()

        mailObj.send()

"""
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pickle as pkl
import numpy as np
import smtplib
import socket
import copy
import json
import os


class MailCls(object):
    """This class is used to manage the sending of the notification after a completed test run."""

    def __init__(self):
        """Initialization of MailCls."""
        self.attr = dict()

        # Constitutive attributes
        self.attr['subject'] = None

        self.attr['message'] = None

        self.attr['attachment'] = None

        # Setup
        self.attr['sender'] = socket.gethostname()

        self.attr['recipient'] = 'eisenhauer@policy-lab.org'

        # Derived attributes
        self.attr['username'] = None

        self.attr['password'] = None

        # Status indicator
        self.is_locked = False

    def send(self):
        """Send message."""
        # Antibugging
        np.testing.assert_equal(self.get_status(), True)

        # Distribute class attributes
        subject = self.attr['subject']

        message = self.attr['message']

        sender = self.attr['sender']

        recipient = self.attr['recipient']

        username = self.attr['username']

        password = self.attr['password']

        attachment = self.attr['attachment']

        # Connect to gmail
        try:
            server = smtplib.SMTP('smtp.gmail.com:587')
        except socket.gaierror:
            return

        server.starttls()

        server.login(username, password)

        # Formatting
        msg = MIMEMultipart('alternative')

        msg['Subject'], msg['From'] = subject, sender

        # Attachment
        if attachment is not None:
            f = open(attachment, 'r')

            attached = MIMEText(f.read())

            attached.add_header('Content-Disposition', 'attachment', filename=attachment)

            msg.attach(attached)

        # Message
        message = MIMEText(message, 'plain')

        msg.attach(message)

        # Send
        server.sendmail(sender, recipient, msg.as_string())

        # Disconnect
        server.quit()

    def _derived_attributes(self):
        """Construct derived attributes."""
        # Antibugging
        np.testing.assert_equal(self.get_status(), True)

        # Check availability
        np.testing.assert_equal(self.attr['message'] is not None, True)

        # Process credentials
        dict_ = json.load(open(os.environ['HOME'] + '/.credentials'))

        self.attr['username'] = dict_['username']

        self.attr['password'] = dict_['password']

    def get_status(self):
        """Get status of class instance."""
        return self.is_locked

    def lock(self):
        """Lock class instance."""
        # Antibugging
        np.testing.assert_equal(self.get_status(), False)

        # Update class attributes
        self.is_locked = True

        # Finalize
        self._derived_attributes()

        self._check_integrity()

    def unlock(self):
        """Unlock class instance."""
        # Antibugging
        np.testing.assert_equal(self.get_status(), True)

        # Update class attributes
        self.is_locked = False

    def get_attr(self, key, deep=False):
        """Get attributes."""
        # Antibugging
        np.testing.assert_equal(self.get_status(), True)
        np.testing.assert_equal(deep in [True, False], True)

        # Copy requested object
        if deep:
            attr = copy.deepcopy(self.attr[key])
        else:
            attr = self.attr[key]

        # Finishing.
        return attr

    def set_attr(self, key, value, deep=False):
        """Get attributes."""
        # Antibugging
        np.testing.assert_equal(self.get_status(), False)
        np.testing.assert_equal(key in self.attr.keys(), True)

        # Copy requested object
        if deep:
            attr = copy.deepcopy(value)
        else:
            attr = value

        # Finishing
        self.attr[key] = attr

    @staticmethod
    def _check_integrity():
        """Check integrity of class instance."""

    def store(self, file_name):
        """Store class instance."""
        # Antibugging
        np.testing.assert_equal(self.get_status(), True)
        np.testing.assert_equal(isinstance(file_name, str), True)

        # Store
        pkl.dump(self, open(file_name, 'wb'))
