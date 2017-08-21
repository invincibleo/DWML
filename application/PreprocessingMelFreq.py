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

from core.PreProcessing import Preprocessing
from core.features import *
from core.util import *


class PreprocessingMelFreq(Preprocessing):
    def extract(self):
        feature_files = []
        feature_extractor = FeatureExtractor(overwrite=False, store=False)
        how_many_bottlenecks = 0
        in_data_list = self.in_dataset.get_data_files()
        out_dataset_list = {}
        for label_name, label_lists in in_data_list.items():
            out_dataset_list[label_name] = {'subdir': '', 'validation': [], 'testing': [], 'training': []}
            for category in ['training', 'testing', 'validation']:
                category_list = label_lists[category]
                for index, data_name in enumerate(category_list):
                    audio_file = get_data_file_path(self.in_dataset, label_name, data_name, self.in_dataset.get_dataset_dir())
                    # Load audio with correct parameters
                    y, fs = AudioFile().load(filename=audio_file, mono=True, fs=44100)

                    y = numpy.reshape(y, [1, -1])

                    for channel in range(0, y.shape[0]):
                        buf = y[channel]
                        mean_value = numpy.mean(buf)
                        buf -= mean_value
                        max_value = max(abs(buf)) + 0.005
                        y[channel] = buf / max_value

                    feature = {}
                    for i in range(10):
                        frame_start = int(i * 0.96 * fs)
                        fram_end = int((i + 1) * 0.96 * fs) - 1

                        # some audio files have duration less than 10sec
                        if frame_start > y.shape[1] or fram_end > y.shape[1]:
                            break
                        raw_audio = y[:, frame_start:fram_end]
                        feature_data = feature_extractor.extract(audio_file=raw_audio)
                        feature_data = feature_data.get_path('mel.feat')
                        feature_data = numpy.reshape(feature_data, (1, 96 * 64))
                        feature[i] = numpy.squeeze(feature_data, 0)


                    output_file_path = get_data_file_path(self.in_dataset, label_name, data_name, self.FLAGS.bottleneck_dir) + '.txt'
                    ensure_dir_exists(self.FLAGS.bottleneck_dir)
                    with open(output_file_path, "w") as melFile:
                        for i in range(10):
                            melFile.write(feature[i])

                    out_dataset_list[label_name][category].append(data_name + '.txt')
                    how_many_bottlenecks += 1
                    if how_many_bottlenecks % 100 == 0:
                        print(str(how_many_bottlenecks) + ' Melfreq files created.')

        return feature_files
