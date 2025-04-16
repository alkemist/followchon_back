from django.db import models
from django.utils.translation import gettext_lazy as _


class Log_Level(models.TextChoices):
    LOCAL = 'local', _('Local')
    INFO = 'info', _('Info')
    EVENT = 'event', _('Event')
    STATISTIC = 'statistic', _('Statistic')
    WARNING = 'warning', _('Warning')
    HOT = 'hot', _('Hot')
    ERROR = 'error', _('Error')
    FAIL = 'fail', _('Fail')
