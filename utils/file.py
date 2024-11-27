import os
import re

import numpy

from .array import ArrayHelper


class FileHelper:

    @staticmethod
    def list_files(path, regex):
        return numpy.sort([
            f
            for f
            in os.listdir(path)
            if re.search(regex, f)
        ])

    @staticmethod
    def read_lines(file_path):
        return numpy.sort(ArrayHelper.clean_list(
            open(file_path, "r").read().split("\n")
        ))

    @staticmethod
    def convert_size(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(numpy.floor(numpy.log(size_bytes, 1024)))
        p = pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])
