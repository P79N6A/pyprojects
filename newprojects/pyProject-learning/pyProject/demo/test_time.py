# -*- coding: utf-8 -*-
import time
from datetime import datetime


def strtime_to_datetime(timestr):
    """
    :param timestr: {str}'2016-02-25 20:21:04.242'
    :return: {datetime}2016-02-25 20:21:04.242000
    """
    local_datetime = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S")
    return local_datetime


def datetime_to_timestamp(datetime_obj):
    local_timestamp = (time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
    return local_timestamp

local_datetime = strtime_to_datetime("2018-07-24 15:10:31")
timestamp = long(datetime_to_timestamp(local_datetime))
print(timestamp)