from django.db import models
from django.utils.translation import gettext_lazy as _


class Architecture(models.TextChoices):
    HAILO = 'hailo', _('Hailo')
    NATIF = 'natif', _('Natif')
