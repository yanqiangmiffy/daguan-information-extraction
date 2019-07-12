#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: submit.py 
@time: 2019-07-12 20:40
@description:
"""
import os

os.system("python 001.make_submit.py "
          "-submit_file data/sub_test_ner.txt "
          "-predict_file submit/a_submit_test_ner_predict_1.txt"
          )
