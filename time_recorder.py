#!/usr/bin/env python
# encoding: utf-8

"""
@author: czhang
@contact: 
@Time : 2019/6/11 21:20
"""

import time
import logging
from contextlib import contextmanager
level = logging.DEBUG
logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)


@contextmanager
def time_record_(message, print_=False):
    """
    :param message: ori function desc
    :param print_: print the logging or not
    :return:
    """
    start_time = time.time()
    start_str = "['{}'] --- 'begin':[{}]]".format(message, time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                  time.localtime(start_time)))
    logging.info(start_str)
    if print_:
        print(start_str)
    yield
    end_time = time.time()
    end_str = "['{}'] --- 'end':[{}]] --- 'last for':[{:.2f}]s".format(message, time.strftime("%Y-%m-%d %H:%M:%S",
                                                                                  time.localtime(end_time)), end_time - start_time)
    logging.info(end_str)
    if print_:
        print(end_str)


if __name__ == '__main__':
    with time_record_("test time recoder", print_=False):
        res = 0
        for i in range(1000):
            for j in range(1000):
                res += j
