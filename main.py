#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 07.08.17 16:15
# @Author  : Duowei Tang
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : main
# @Software: PyCharm Community Edition

from application.Youtube8MDataset import *
from core import *

def main():
    dataset = Youtube8MDataset('/media/invincibleo/Windows/Users/u0093839/Leo/Audioset', 10, 10, ['wav', 'mp3'])
    dataset.get_training_files()
    print('a')


if __name__ == '__main__':
    main()