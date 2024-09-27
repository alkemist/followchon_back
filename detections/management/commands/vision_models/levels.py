from django.db import models
from django.utils.translation import gettext_lazy as _


class Levels(models.TextChoices):
    INFO = 'info', _('Info')
    EVENT = 'event', _('Event')
    STATISTIC = 'statistic', _('Statistic')
    WARNING = 'warning', _('Warning')
    TEMP = 'temp', _('Temp')
    ERROR = 'error', _('Error')
    FAIL = 'fail', _('Fail')
