#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 08.08.17 15:22
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : PreprocessingMelFreq
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

from core.PreProcessing import Preprocessing

class PreprocessingMelFreq(Preprocessing):
    def extract(self):
        feature_files = []
        feature_extractor = self.FeatureExtractor(overwrite=overwrite, store=True)
        for file_id, audio_filename in enumerate(tqdm(files,
                                                      desc='           {0:<15s}'.format('Extracting features '),
                                                      file=sys.stdout,
                                                      leave=False,
                                                      disable=self.disable_progress_bar,
                                                      ascii=self.use_ascii_progress_bar
                                                      )):

            # Get feature filename
            current_feature_files = self._get_feature_filename(audio_file=os.path.split(audio_filename)[1],
                                                               path=self.params.get_path('path.feature_extractor'))

            if not filelist_exists(current_feature_files) or overwrite:
                feature_extractor.extract(
                    audio_file=self.dataset.relative_to_absolute_path(audio_filename),
                    extractor_params=DottedDict(self.params.get_path('feature_extractor.parameters')),
                    storage_paths=current_feature_files
                )

            feature_files.append(current_feature_files)
        return feature_files