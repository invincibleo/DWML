#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21/08/2017 3:30 PM
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : test_Dataset
# @Software: PyCharm Community Edition

import unittest

from application.Youtube8MDataset import Youtube8MDataset

class test_Dataset(unittest.TestCase):
    def test_save_dataset(self):
        dataset = Youtube8MDataset(None, 0.1, 0.1)
        
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
