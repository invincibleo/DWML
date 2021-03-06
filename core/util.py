#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 08.08.17 14:51
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : util
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy
import tensorflow as tf

def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.
    
    Args:
    dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_data_file_path(dataset, label_name, data_name, base_dir):
    data_list = dataset.get_data_files()
    if label_name not in data_list.keys():
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = data_list[label_name]

    sub_dir = label_lists['subdir']
    full_path = os.path.join(base_dir, sub_dir, data_name)
    return full_path



