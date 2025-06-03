import random
import re
from datetime import datetime
from pathlib import Path


class DateHelper:

    @staticmethod
    def secondsToMMSS(seconds: float):
        minutes, seconds = divmod(round(seconds), 60)
        return '%d:%02d' % (minutes, seconds)

    @staticmethod
    def filenameToDate(path: str):
        file_date = Path(path).stem
        date_values = re.split('[-_]', file_date)
        return datetime(
            int(date_values[0]),
            int(date_values[1]),
            int(date_values[2]),
            int(date_values[3]),
            int(date_values[4]),
            int(date_values[5]),
            random.randint(0, 500)
        )
