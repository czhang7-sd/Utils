#!/usr/bin/env python
# encoding: utf-8

"""
@author: czhang
@contact:
@Time : 2020/1/21 12:07
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
