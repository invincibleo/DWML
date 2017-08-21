#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 08.08.17 14:20
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : PreprocessingInceptionV3
# @Software: PyCharm Community Edition


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import numpy
import tensorflow as tf
import cPickle as pickle

from six.moves import urllib
from scipy.misc import imsave


from core.PreProcessing import Preprocessing
from core.Dataset import Dataset
from core.util import *
from core.GeneralReader import GeneralReader
from application.Youtube8MDataset import Youtube8MDataset

class PreprocessingInceptionV3(Preprocessing):
    def extract(self):
        self.maybe_download_and_extract()
        graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = self.create_inception_graph()

        with tf.Session(graph=graph) as sess:
            self.cache_bottlenecks(sess, jpeg_data_tensor, bottleneck_tensor)
        return self.out_dataset

    def maybe_download_and_extract(self):
        """Download and extract model tar file.

        If the pretrained model we're using doesn't already exist, this function
        downloads it from the TensorFlow.org website and unpacks it into a directory.
        """
        DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        dest_directory = self.FLAGS.model_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' %
                                 (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def create_inception_graph(self):
        """"Creates a graph from saved GraphDef file and returns a Graph object.
  
        Returns:
          Graph holding the trained Inception network, and various tensors we'll be
          manipulating.
        """
        BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
        JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
        RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

        with tf.Graph().as_default() as graph:
            model_filename = os.path.join(
                self.FLAGS.model_dir, 'classify_image_graph_def.pb')
            with tf.gfile.FastGFile(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                    tf.import_graph_def(graph_def, name='', return_elements=[
                        BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                        RESIZED_INPUT_TENSOR_NAME]))
        return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

    def cache_bottlenecks(self, sess, jpeg_data_tensor, bottleneck_tensor):

        """Ensures all the training, testing, and validation bottlenecks are cached.

        Because we're likely to read the same image multiple times (if there are no
        distortions applied during training) it can speed things up a lot if we
        calculate the bottleneck layer values once for each image during
        preprocessing, and then just read those cached values repeatedly during
        training. Here we go through all the images we've found, calculate those
        values, and save them off.

        Args:
        sess: The current active TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        image_dir: Root folder string of the subfolders containing the training
        images.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        jpeg_data_tensor: Input tensor for jpeg data from file.
        bottleneck_tensor: The penultimate output layer of the graph.

        Returns:
        Nothing.
        """
        how_many_bottlenecks = 0
        ensure_dir_exists(self.FLAGS.bottleneck_dir)
        out_dataset_list = {}
        if not tf.gfile.Exists('out_dataset.pickle'):
            in_data_list = self.in_dataset.get_data_files()
            for label_name, label_lists in in_data_list.items():
                out_dataset_list[label_name] = {'subdir': '', 'validation': [], 'testing': [], 'training': []}
                for category in ['training', 'testing', 'validation']:
                    category_list = label_lists[category]
                    for index, data_name in enumerate(category_list):
                        self.get_or_create_bottleneck(sess, self.in_dataset, label_name, data_name, self.FLAGS.bottleneck_dir,
                                                 jpeg_data_tensor, bottleneck_tensor)
                        out_dataset_list[label_name][category].append(data_name + '.txt')
                        how_many_bottlenecks += 1
                        if how_many_bottlenecks % 100 == 0:
                            print(str(how_many_bottlenecks) + ' bottleneck files created.')

            self.out_dataset = Dataset(name='bottleneck_dataset', dataset_dir=self.FLAGS.bottleneck_dir, data_list=out_dataset_list)
            pickle.dump(self.out_dataset, open('out_dataset.pickle', 'wb'), protocol=2)  # pickle.HIGHEST_PROTOCOL)
        else:
            self.out_dataset = pickle.load(open('out_dataset.pickle', "rb"))

    def get_or_create_bottleneck(self, sess, dataset, label_name, data_name, bottleneck_dir, jpeg_data_tensor,
                                 bottleneck_tensor):
        """Retrieves or calculates bottleneck values for an image.

        If a cached version of the bottleneck data exists on-disk, return that,
        otherwise calculate the data and save it to disk for future use.

        Args:
        sess: The current active TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        label_name: Label string we want to get an image for.
        index: Integer offset of the image we want. This will be modulo-ed by the
        available number of images for the label, so it can be arbitrarily large.
        image_dir: Root folder string  of the subfolders containing the training
        images.
        category: Name string of which  set to pull images from - training, testing,
        or validation.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        jpeg_data_tensor: The tensor to feed loaded jpeg data into.
        bottleneck_tensor: The output tensor for the bottleneck values.

        Returns:
        Numpy array of values produced by the bottleneck layer for the image.
        """
        label_lists = dataset.get_data_files()[label_name]
        sub_dir = label_lists['subdir']
        sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
        ensure_dir_exists(sub_dir_path)

        bottleneck_path = self.get_bottleneck_path(dataset, label_name, data_name,
                                              bottleneck_dir)
        if not os.path.exists(bottleneck_path):
            self.create_bottleneck_file(bottleneck_path, dataset, label_name,
                                        data_name, sess, jpeg_data_tensor, bottleneck_tensor)

        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read().splitlines()

        return bottleneck_string

    def get_bottleneck_path(self, dataset, label_name, data_name, bottleneck_dir):
        """"Returns a path to a bottleneck file for a label at the given index.
  
        Args:
          image_lists: Dictionary of training images for each label.
          label_name: Label string we want to get an image for.
          index: Integer offset of the image we want. This will be moduloed by the
          available number of images for the label, so it can be arbitrarily large.
          bottleneck_dir: Folder string holding cached files of bottleneck values.
          category: Name string of set to pull images from - training, testing, or
          validation.
  
        Returns:
          File system path string to an image that meets the requested parameters.
        """
        return get_data_file_path(dataset, label_name, data_name, bottleneck_dir) + '.txt'

    def create_bottleneck_file(self, bottleneck_path, dataset, label_name, data_name,
                               sess, jpeg_data_tensor, bottleneck_tensor):
        """Create a single bottleneck file."""
        print('Creating bottleneck at ' + bottleneck_path)
        data_path = get_data_file_path(dataset, label_name, data_name, dataset.get_dataset_dir())
        if not tf.gfile.Exists(data_path):
            tf.logging.fatal('File does not exist %s', data_path)
        data_content = GeneralReader.read(data_path)

        bottleneck_string_total = ''
        for key, image_data in data_content.iteritems():
            imsave('buf.jpeg', image_data)
            image_data = tf.gfile.FastGFile('buf.jpeg', 'rb').read()
            try:
                bottleneck_values = self.run_bottleneck_on_image(
                    sess, image_data, jpeg_data_tensor, bottleneck_tensor)
            except:
                raise RuntimeError('Error during processing file %s' % data_path)

            bottleneck_string = ','.join(str(x) for x in bottleneck_values)
            bottleneck_string_total = bottleneck_string_total + bottleneck_string + '\n'

        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string_total)

    def run_bottleneck_on_image(self, sess, image_data, image_data_tensor,
                                bottleneck_tensor):
        """Runs inference on an image to extract the 'bottleneck' summary layer.
  
        Args:
          sess: Current active TensorFlow Session.
          image_data: String of raw JPEG data.
          image_data_tensor: Input data layer in the graph.
          bottleneck_tensor: Layer before the final softmax.
  
        Returns:
          Numpy array of bottleneck values.
        """
        bottleneck_values = sess.run(
            bottleneck_tensor,
            {image_data_tensor: image_data})
        bottleneck_values = numpy.squeeze(bottleneck_values)
        return bottleneck_values


