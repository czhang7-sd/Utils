#!/usr/bin/env python
# encoding: utf-8

"""
@Email: zhangcan895@pingan.com.cn
@Time : 2020/3/4 14:05
"""

import os

__env = os.environ.get("env", "DEV")

__kafka_man = {
    "DEV": {
        "host": "localhost",
        "port": 9092
    },

    "TEST":{},
    "UAT": {},
    "PROD": {}
}

kafka_configuration = __kafka_man[__env]
