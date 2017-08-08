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


class Dataset(object):
    """A simple class for handling data sets."""

    def __init__(self, name, dataset_dir, data_list=None):
        """Initialize dataset using a subset and the path to the data."""
        self.name = name
        if data_list is None:
            self.data_list = self.create_data_list()
        else:
            self.data_list = data_list
        self.dataset_dir = dataset_dir
        self.num_classes = len(self.data_list.keys())

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return self.num_classes
        # return 10

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation']

    def create_data_list(self):
        return -1

    def get_dataset_dir(self):
        return self.dataset_dir

    def get_data_files(self):
        """Returns a python list of all (sharded) data subset files.
        Returns:
          python list of all (sharded) data set files.
        Raises:
          ValueError: if there are not data_files matching the subset.
        """
        return self.data_list

    def get_training_files(self):
        training_list = {}
        for keys in self.data_list:
            training_list[keys] = self.data_list[keys]['training']
        return training_list

    def get_testing_files(self):
        testing_list = {}
        for keys in self.data_list:
            testing_list[keys] = self.data_list[keys]['testing']
        return testing_list

    def get_validation_files(self):
        validation_list = {}
        for keys in self.data_list:
            validation_list[keys] = self.data_list[keys]['validation']
        return validation_list