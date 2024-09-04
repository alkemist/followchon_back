class DateHelper:

    @staticmethod
    def secondsToMMSS(seconds: float):
        minutes, seconds = divmod(round(seconds), 60)
        return '%d:%02d' % (minutes, seconds)
