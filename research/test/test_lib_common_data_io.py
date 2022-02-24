#!/usr/bin/env python
# -*- coding:gb18030 -*-
#Author:   zhanghao55@baidu.com
#Date  :   20/04/01 19:27:05

import os
import sys
import unittest
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../" % _cur_dir)

import logging
from lib.common.logger import init_log, set_level
init_log("log/test_lib_common_data_io", stream_level=logging.INFO)

import lib.common.data_io as data_io

test_data_root = "test/data/test_lib_common_data_io/"
test_data_dir = os.path.join(test_data_root, "data_dir1")
test_list_file =  os.path.join(test_data_root, "test_list.txt")
test_list_pkl =  os.path.join(test_data_root, "test_list.pkl")

test_list = [
        [1, "key1", 0.5],
        [2, "key2", 0.6],
        [3, "key3", 0.7],
        ]


class TestSend(unittest.TestCase):

    def test_1_get_file_name_list(self):
        file_list = data_io.get_file_name_list(test_data_dir)
        self.assertEqual(len(file_list), 5)

    def test_2_get_data(self):
        data = list(data_io.get_data(test_data_dir, encoding="utf-8"))
        self.assertEqual(len(data), 24932)

    def test_3_write_to_file(self):
        def to_str(cur_list):
            return "{:d}\t{}\t{:.2f}".format(cur_list[0], cur_list[1], cur_list[2])
        data_io.write_to_file(test_list, test_list_file, write_func=to_str)

    def test_4_read_from_file(self):
        def to_list(cur_str):
            parts = cur_str.strip("\n").split("\t")
            return int(parts[0]), parts[1], float(parts[2])
        read_list = data_io.read_from_file(test_list_file, read_func=to_list)
        self.assertTrue(check_equal(read_list, test_list))

    def test_5_dump_pkl(self):
        data_io.dump_pkl(test_list, test_list_pkl, overwrite=True)

    def test_6_load_pkl(self):
        load_list = data_io.load_pkl(test_list_pkl)
        self.assertTrue(check_equal(load_list, test_list))


def check_equal(src_list, tar_list):
    for cur_src_list, cur_tar_list in zip(src_list, tar_list):
        for cur_src_ele, cur_tar_ele in zip(cur_src_list, cur_tar_list):
            if cur_src_ele != cur_tar_ele:
                return False
    return True


if __name__ == "__main__":
    unittest.main()

