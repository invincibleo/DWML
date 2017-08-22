#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21/08/2017 3:45 PM
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : DCASE2017Task3Dataset
# @Software: PyCharm Community Edition


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.Dataset import Dataset

import os
import csv
import re
import hashlib
import os.path
import cPickle as pickle
import numpy
import random

from tqdm import tqdm
from itertools import islice
from core.features import *
from core.util import *

import tensorflow as tf


class DCASE2017Task3Dataset(Dataset):
    def __init__(self, dataset_dir, testing_percentage, validation_percentage, extensions=['wav', 'mp3']):
        """Initialize dataset using a subset and the path to the data."""

        self.name = 'DCASE2017Task3'
        self.num_classes = 43
        self.dataset_dir = dataset_dir
        self.testing_percentage = testing_percentage
        self.validation_percentage = validation_percentage
        self.extensions = extensions
        self.max_num_data_per_class = 59

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation', 'test']

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
