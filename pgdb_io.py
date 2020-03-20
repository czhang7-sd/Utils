#!/usr/bin/env python
# encoding: utf-8

"""
@author: czhang
@contact: 
@Time : 2020/3/20 0020 17:15 
"""
import psycopg2
import pandas.io.sql as sqlio

class PgDbIo():
    def __init__(self, **kwargs):
        super(PgDbIo, self).__init__()
        try:
            self.host = kwargs.get("host", "localhost")
            self.port = kwargs.get("port", 5432)
            self.database = kwargs.get("database", "utils")
            self.username = kwargs.get("user", "admin")
            self.password = kwargs.get("password", "123")

        except Exception as e:
            raise Exception("lack the necessary params".format(e))

        try:
            self.connect_str = """dbname={} user='{}' host='{}' password={}""".format(self.database, self.username, self.host, self.password, self.port)
            self.conn = psycopg2.connect(self.connect_str)
        except Exception as e:
            raise Exception("failed connecting the database".format(e))

    def find_data(self, condition, limit=None):
        table = condition.get("table", "log")
        fields = condition.get("fields", "*")
        start_time = condition.get("start_time", "2019-12-12")
        end_time = condition.get("end_time", "2019-12-13")
        try:
            sql = (r"""select {} from {} where created_time >= '{}' and created_time < '{}' """).format(fields, table, start_time, end_time)
            df = sqlio.read_sql_query(sql, self.conn)
        except Exception as e:
            raise e
        return df

    def insert_data(self, table_name, fields, values):
        try:
            cursor = self.conn.cursor()
            assert len(fields) == len(values)
            num = len(fields)

            field_str = r" (" + ','.join(fields) + r") "
            bracket_str = r""", """.join([r"{}" for i in range(num)])
            values_conv = [r"""'""" + i + r"""'""" if type(i) == str else i for i in values]
            values_str = bracket_str.format(*values_conv)

            



