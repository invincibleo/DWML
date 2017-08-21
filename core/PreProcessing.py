#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 08.08.17 14:13
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : PreProcessing
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod


class Preprocessing(object):
    __metaclass__ = ABCMeta

    def __init__(self, FLAGS, in_dataset):
        self.in_dataset = in_dataset
        self.out_dataset = None
        self.if_cache = False
        self.FLAGS = FLAGS

    def set_if_cache(self, boolean):
        self.if_cache = boolean

    @abstractmethod
    def extract(self):
        pass

    def get_out_dataset(self):
        if self.out_dataset is not None:
            return self.out_dataset
        else:
            self.extract()
            return self.out_dataset