from django.db import models
from django.utils.translation import gettext_lazy as _


class Event_Type(models.TextChoices):
    AGENT_LOG = 'agent_log', _('Agent log')
    BRAIN_PROCESS = 'brain_process', _('Brain process')
