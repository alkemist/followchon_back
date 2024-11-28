from django.db import models
from django.utils.translation import gettext_lazy as _


class Source(models.TextChoices):
    VISION = 'vision', _('Vision')
    VIDEO = 'video', _('Video')
    PHOTO = 'photo', _('Photo')
