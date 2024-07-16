import numpy
import os
import re

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
