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
from core.GeneralFileAccessor import GeneralFileAccessor

import tensorflow as tf


class DCASE2017Task3Dataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.name = 'DCASE2017Task3Dataset'
        self.num_classes = 43
        self.dataset_dir = kwargs.get('dataset_dir')
        self.max_num_data_per_class = 59
        self.data_list = kwargs.get('data_list')
        self.testing_percentage = kwargs.get('testing_percentage')
        self.validation_percentage = kwargs.get('validation_percentage')
        self.training_percentage = kwargs.get('training_percentage')
        self.extensions = kwargs.get('extensions')

        super(DCASE2017Task3Dataset, self).__init__(name=self.name, dataset_dir=self.dataset_dir, data_list=self.data_list)


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
        meta_file_addr = os.path.join(self.dataset_dir, 'meta.txt')
        meta_content = GeneralFileAccessor(meta_file_addr).read()

        data_list = {}
        line_idx = 0
        audio_frame_size = 1
        for line in meta_content:
            line_list = line.split('\t')
            file_name = line_list[0][6:]
            sub_dir = 'audio'
            start_time = float(line_list[2])
            end_time = float(line_list[3])
            label_name = line_list[4]
            duration = end_time - start_time
            if duration >= audio_frame_size:
                i = 0
                while (int(duration / audio_frame_size)):
                    time_idx = [start_time + i*audio_frame_size, start_time + (i+1)*audio_frame_size]
                    duration = duration - audio_frame_size

                    hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(file_name+str(line_idx)+str(i))).hexdigest()
                    percentage_hash = ((int(hash_name_hashed, 16) % (self.max_num_data_per_class + 1)) *
                                       (100.0 / self.max_num_data_per_class))

                    if not label_name in data_list.keys():
                        data_list[label_name] = {'subdir':sub_dir, 'validation':[], 'testing':[], 'training':[]}

                    if percentage_hash < self.validation_percentage:
                        data_list[label_name]['validation'].append((file_name, time_idx))
                    elif percentage_hash < (self.testing_percentage + self.validation_percentage):
                        data_list[label_name]['testing'].append((file_name, time_idx))
                    else:
                        data_list[label_name]['training'].append((file_name, time_idx))

                    i = i + 1

                # time_idx = [start_time + i * audio_frame_size, end_time]
                #
                # hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(file_name + str(line_idx) + str(i))).hexdigest()
                # percentage_hash = ((int(hash_name_hashed, 16) % (self.max_num_data_per_class + 1)) *
                #                    (100.0 / self.max_num_data_per_class))
                #
                # if not label_name in data_list.keys():
                #     data_list[label_name] = {'subdir': sub_dir, 'validation': [], 'testing': [], 'training': []}
                #
                # if percentage_hash < self.validation_percentage:
                #     data_list[label_name]['validation'].append((file_name, time_idx))
                # elif percentage_hash < (self.testing_percentage + self.validation_percentage):
                #     data_list[label_name]['testing'].append((file_name, time_idx))
                # else:
                #     data_list[label_name]['training'].append((file_name, time_idx))
            #
            # else:
            #     hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(file_name+str(line_idx))).hexdigest()
            #     percentage_hash = ((int(hash_name_hashed, 16) % (self.max_num_data_per_class + 1)) *
            #                        (100.0 / self.max_num_data_per_class))
            #
            #     if not label_name in data_list.keys():
            #         data_list[label_name] = {'subdir': sub_dir, 'validation': [], 'testing': [], 'training': []}
            #
            #     if percentage_hash < self.validation_percentage:
            #         data_list[label_name]['validation'].append((file_name, [start_time, end_time]))
            #     elif percentage_hash < (self.testing_percentage + self.validation_percentage):
            #         data_list[label_name]['testing'].append((file_name, [start_time, end_time]))
            #     else:
            #         data_list[label_name]['training'].append((file_name, [start_time, end_time]))
            line_idx = line_idx + 1

        return data_list

    def generate_arrays_from_file_a(self, category, batch_size=100):
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
            data_name = working_list[data_idx][0][0]
            time_span = working_list[data_idx][0][-1]
            label_idx = working_list[data_idx][1]
            first_label = int(label_idx.split(',')[0])
            label_name = classes_ordered[first_label]

            bottleneck_path = get_data_file_path(self, label_name, data_name, self.dataset_dir)

            bottleneck = self.extract(bottleneck_path, time_span) ######### xin jia
            bottleneck = bottleneck.values()
            bottleneck = numpy.reshape(bottleneck, (-1, 1, 96*20))
            bottleneck = numpy.squeeze(bottleneck, axis=1)
            if bottleneck.shape[0] <= 0:
                continue

            # bottleneck = numpy.concatenate((bottleneck, numpy.zeros((bottleneck.shape[0], 140 - bottleneck.shape[1]))), axis=1)
            # bottleneck = numpy.concatenate((bottleneck, numpy.zeros((140 - bottleneck.shape[0], bottleneck.shape[1]))), axis=0)


            hot_label = numpy.zeros(self.num_classes, numpy.int8)
            for idx in label_idx.split(','):
                hot_label[int(idx)] = 1

            if not len(X) and not len(Y):
                X = bottleneck
                Y = numpy.matlib.repmat(hot_label, m=bottleneck.shape[0], n=1) #numpy.size(bottleneck, 0)/96
            else:
                X = numpy.append(X, bottleneck, 0)
                Y = numpy.append(Y, numpy.matlib.repmat(hot_label, m=bottleneck.shape[0], n=1), 0)

            try:
                if X.shape[0] >= batch_size:
                    X = numpy.reshape(X, (X.shape[0], -1, 20))  #######you wen ti
                    X = numpy.expand_dims(X, axis=3)
                    Y = numpy.reshape(Y, (-1, self.num_classes))
                    if X.shape[0] >= batch_size and Y.shape[0] >= batch_size:
                        yield (X, Y)
                        i = 0
                        X = []
                        Y = []
            except:
                i = 0
                X = []
                Y = []



    def extract(self, audio_file, time_span):
        feature_extractor = FeatureExtractor(overwrite=False, store=False)
        start_time = time_span[0]
        end_time = time_span[1]
        y, fs = AudioFile().load(filename=audio_file, mono=True, fs=44100, start=start_time, stop=end_time)

        y = numpy.reshape(y, [1, -1])

        for channel in range(0, y.shape[0]):
            buf = y[channel]
            mean_value = numpy.mean(buf)
            buf -= mean_value
            max_value = max(abs(buf)) + 0.005
            y[channel] = buf / max_value

        # feature_data = feature_extractor.extract(audio_file=y)
        # feature_data = feature_data.get_path('mel.feat')
        # feature = numpy.reshape(feature_data, (-1, 64))[:960, :]

        feature = {}
        for i in range(10):
            frame_start = int(i * 0.96 * fs)
            fram_end = int((i + 1) * 0.96 * fs) - 1

            # some audio files have duration less than 10sec
            if frame_start > y.shape[1] or fram_end > y.shape[1]:
                break
            raw_audio = y[:, frame_start:fram_end]
            feature_data = feature_extractor.extract(audio_file=raw_audio)
            feature_data = feature_data.get_path('mfcc.feat')
            feature_data = numpy.reshape(feature_data, (1, 96 * 20))
            # feature = feature + numpy.squeeze(feature_data, 0)
            # feature[i] = feature_data
            feature[i] = feature_data
        return feature

    def get_k_hot_label(self):
        meta_file_addr = os.path.join(self.dataset_dir, 'meta.txt')
        meta_content = GeneralFileAccessor(meta_file_addr).read()

        for line_content in meta_content:
            line_list = line_content.split('\t')
