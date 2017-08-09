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

import random
import numpy

from core.util import *


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

    def get_num_classes(self):
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
        training_list = []
        for keys in self.data_list:
            training_list = training_list + self.data_list[keys]['training']
        return training_list

    def get_testing_files(self):
        testing_list = []
        for keys in self.data_list:
            testing_list = testing_list + self.data_list[keys]['testing']
        return testing_list

    def get_validation_files(self):
        validation_list = []
        for keys in self.data_list:
            validation_list = validation_list + self.data_list[keys]['validation']
        return validation_list

    def generate_arrays_from_file(self, category, batch_size=100):
        i = 0
        X = []
        Y = []
        if category == 'training':
            working_list = self.get_training_files()
        elif category == 'validation':
            working_list = self.get_validation_files()
        elif category == 'testing':
            working_list = self.get_testing_files()

        num_data_files = len(working_list)
        classes_ordered = [x for x in self.data_list.keys()]
        classes_ordered = numpy.sort(classes_ordered)
        while (1):
            data_idx = random.randrange(num_data_files)
            data_name = working_list[data_idx][0]
            label_idx = working_list[data_idx][1]
            first_label = int(label_idx.split(',')[0])
            label_name = classes_ordered[first_label]

            bottleneck_path = get_data_file_path(self, label_name, data_name, self.dataset_dir)

            bottleneck = numpy.loadtxt(bottleneck_path, delimiter=',')
            bottleneck = numpy.reshape(bottleneck, (-1, 2048)) ##########you wen ti 2048

            hot_label = numpy.zeros(self.num_classes, numpy.int8)
            for idx in label_idx.split(','):
                hot_label[int(idx)] = 1

            if not len(X) and not len(Y):
                X = bottleneck
                Y = numpy.matlib.repmat(hot_label, m=numpy.size(bottleneck, 0), n=1)
            else:
                X = numpy.append(X, bottleneck, 0)
                Y = numpy.append(Y, numpy.matlib.repmat(hot_label, m=numpy.size(bottleneck, 0), n=1), 0)

            if len(X) >= batch_size:
                X = numpy.reshape(X, (-1, 2048))  #######you wen ti
                Y = numpy.reshape(Y, (-1, self.num_classes))
                if len(X) and len(Y):
                    rand_perm = numpy.random.permutation(batch_size)
                    yield (X[rand_perm, :], Y[rand_perm, :])
                i = 0
                X = []
                Y = []

    def one_hot_encoding(self):
        dataset = self
        data_list = dataset.get_data_files()
        dataset_dir = dataset.get_dataset_dir()
        num_classes = dataset.get_num_classes()
        classes_ordered = [x for x in data_list.keys()]
        classes_ordered = numpy.sort(classes_ordered)

        one_hot_data_list = {}
        for label_name, label_lists in data_list.items():
            one_hot_data_list[label_name] = {'subdir': '', 'training': [], 'testing': [], 'validation': []}
            for category in ['training', 'testing', 'validation']:
                category_list = label_lists[category]
                for index, data_name in enumerate(category_list):
                    idx = classes_ordered.tolist().index(label_name)
                    idx = str(idx)
                    # one_hot_label = numpy.zeros(shape=(1, num_classes))
                    # one_hot_label[-1, idx] = 1
                    one_hot_data_list[label_name][category].append((data_name, idx))

        one_hot_dataset = Dataset('one_hot_encoding_dataset', dataset_dir, one_hot_data_list)
        return one_hot_dataset

    def k_hot_encoding(self):
        dataset = self
        data_list = dataset.get_data_files()
        dataset_dir = dataset.get_dataset_dir()
        classes_ordered = [x for x in data_list.keys()]
        classes_ordered = numpy.sort(classes_ordered)

        k_hot_data_list = {}
        new_data_list = {}
        for label_name, label_lists in data_list.items():
            for category in ['training', 'testing', 'validation']:
                category_list = label_lists[category]
                for index, data_name in enumerate(category_list):
                    if data_name not in new_data_list.keys():
                        new_data_list[data_name] = str(classes_ordered.tolist().index(label_name))
                    else:
                        new_data_list[data_name] = '' + new_data_list[data_name] + ',' + str(classes_ordered.tolist().index(label_name))

        for label_name, label_lists in data_list.items():
            k_hot_data_list[label_name] = {'subdir': '', 'training': [], 'testing': [], 'validation': []}
            for category in ['training', 'testing', 'validation']:
                category_list = label_lists[category]
                for index, data_name in enumerate(category_list):
                    k_hot_data_list[label_name][category].append((data_name, new_data_list[data_name]))

        k_hot_dataset = Dataset('k_hot_encoding_dataset', dataset_dir, k_hot_data_list)
        return k_hot_dataset