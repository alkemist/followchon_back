from django.db import models
from django.utils.translation import gettext_lazy as _


class Type(models.TextChoices):
    ALL = 'all', _('All')
    CHONS = 'chons', _('Chons')
