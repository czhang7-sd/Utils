#!/usr/bin/env python
# encoding: utf-8

"""
@author: czhang
@contact: 
@Time : 2019/9/17 10:13
"""

import numpy as np
import time
import functools


def time_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result_value = func(*args, **kwargs)
        end = time.perf_counter()
        print('{0:<10}.{1:<2} func cost: {2:<8}s'.format(func.__module__, func.__name__, end - start))
        return result_value
    return wrapper


@time_wrapper
def exp_test(v):
    print('test the time wrapper')
    return np.exp(v + 3)


if __name__ == '__main__':
    print(exp_test(3.))
