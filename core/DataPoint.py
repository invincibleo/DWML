#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 08.08.17 10:53
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : DataPoint
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class DataPoint(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    @property
    def value(self):
        return
