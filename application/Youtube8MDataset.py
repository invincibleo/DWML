#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 07.08.17 15:19
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : Youtube_8M_Dataset
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.Dataset import Dataset

from application.ontologyProcessing import OntologyProcessing


import os
import csv
import re
import hashlib
import os.path

from tqdm import tqdm
from itertools import islice
import tensorflow as tf

class Youtube8MDataset(Dataset):
    def __init__(self, dataset_dir, testing_percentage, validation_percentage, extensions=['wav', 'mp3']):
        """Initialize dataset using a subset and the path to the data."""

        self.name = 'Youtube8M'
        self.num_classes = 527
        self.dataset_dir = dataset_dir
        self.testing_percentage = testing_percentage
        self.validation_percentage = validation_percentage
        self.extensions = extensions
        self.max_num_data_per_class = 59

        super(Youtube8MDataset, self).__init__(name=self.name)

        self.data_list = self.create_data_list()

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return self.num_classes

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation', 'test']

    def get_data_files(self):
        """Returns a python list of all (sharded) data subset files.
        Returns:
          python list of all (sharded) data set files.
        Raises:
          ValueError: if there are not data_files matching the subset.
        """
        if self.data_list is None:
            self.data_list = self.create_data_list()
            return self.data_list
        else:
            return self.data_list

    def get_training_files(self):
        training_list = {}
        for keys in self.data_list:
            training_list[keys] = self.data_list[keys]['training']
        return training_list

    def get_testing_files(self):
        training_list = {}
        for keys in self.data_list:
            training_list[keys] = self.data_list[keys]['testing']
        return training_list

    def get_validation_files(self):
        training_list = {}
        for keys in self.data_list:
            training_list[keys] = self.data_list[keys]['validation']
        return training_list

    def create_data_list(self):
        """Builds a list of training images from the file system.
    
        Analyzes the sub folders in the image directory, splits them into stable
        training, testing, and validation sets, and returns a data structure
        describing the lists of images for each label and their paths.
    
        Args:
        image_dir: String path to a folder containing subfolders of images.
        testing_percentage: Integer percentage of the images to reserve for tests.
        validation_percentage: Integer percentage of images reserved for validation.
    
        Returns:
        A dictionary containing an entry for each label subfolder, with images split
        into training, testing, and validation sets within each label.
        """
        if not tf.gfile.Exists(self.dataset_dir):
            print("Dataset directory '" + self.dataset_dir + "' not found.")
            return None
        result = {}
        # The root directory comes first, so skip it.
        file_list = []
        for extension in self.extensions:
            file_glob = os.path.join(self.dataset_dir, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))

        if not file_list:
            print('No files found')
            return None

        aso = OntologyProcessing.get_label_name_list(os.path.join(os.path.dirname(__file__), 'ontology.json'))
        with open(os.path.join(os.path.split(os.path.realpath(__file__))[0], os.path.join(os.path.dirname(__file__), 'balanced_train_segments.csv')), 'rb') as csvfile:
            label_list = csv.reader(csvfile, delimiter=',')

            result = {}
            file_list = [os.path.basename(x) for x in file_list]  # [:-4]
            extension_name = file_list[0].split('.')[-1]
            for label in tqdm(islice(label_list, 3, None), total=22163):
                file_name = label[0] + '.' + extension_name
                file_label = [re.sub(r'[ "]', '', x) for x in label[3:]]

                if file_name in file_list:
                    hash_name = file_name
                    # This looks a bit magical, but we need to decide whether this file should
                    # go into the training, testing, or validation sets, and we want to keep
                    # existing files in the same set even if more files are subsequently
                    # added.
                    # To do that, we need a stable way of deciding based on just the file name
                    # itself, so we do a hash of that and then use that to generate a
                    # probability value that we use to assign it.
                    hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
                    percentage_hash = ((int(hash_name_hashed, 16) % (self.max_num_data_per_class + 1)) *
                                       (100.0 / self.max_num_data_per_class))
                    for label in file_label:
                        label_name = aso[label]['name']
                        if not label_name in result.keys():
                            result[label_name] = {'dir': '', 'validation': [], 'testing': [], 'training': []}

                        if percentage_hash < self.validation_percentage:
                            result[label_name]['validation'].append(file_name)
                        elif percentage_hash < (self.testing_percentage + self.validation_percentage):
                            result[label_name]['testing'].append(file_name)
                        else:
                            result[label_name]['training'].append(file_name)
        return result
