#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 07.08.17 14:22
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Dataset.py
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod


class Dataset(object):
    """A simple class for handling data sets."""
    __metaclass__ = ABCMeta

    def __init__(self, name):
        """Initialize dataset using a subset and the path to the data."""
        self.name = name
        self.data_list = None

    @abstractmethod
    def num_classes(self):
        """Returns the number of classes in the data set."""
        pass
        # return 10

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation']

    @abstractmethod
    def get_data_files(self):
        """Returns a python list of all (sharded) data subset files.
        Returns:
          python list of all (sharded) data set files.
        Raises:
          ValueError: if there are not data_files matching the subset.
        """
        pass
        # return data_list

