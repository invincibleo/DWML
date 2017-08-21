#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 08.08.17 15:54
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : GeneralReader
# @Software: PyCharm Community Edition

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from core.files import *


class GeneralReader(object):
    def __init__(self, file_path):
        self.audio_formats_static = ['wav', 'flac', 'm4a', 'webm']
        self.dict_formats_static = ['json', 'cpickle', 'marshal', 'msgpack']
        self.list_formats_static = ['txt', 'yaml']
        self.extension = file_path.split('.')[-1]
        self.file_path = file_path


    def read(self):
        if self.extension in self.audio_formats_static:
            return AudioFile().load(self.file_path)
        elif self.extension in self.dict_formats_static:
            return DictFile().load(self.file_path)
        elif self.extension in self.list_formats_static:
            return ListFile().load(self.file_path)
        else:
            print('Format does not support!')