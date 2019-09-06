#!/usr/bin/python
# -*- coding:UTF-8 -*-
try:
    raise Exception("Invalid level!", 1)
    fh = open("testing", "w")
    fh.write("testing files")
except IOError:
    print "IOError"
else:
    print "write success"
    fh.close()