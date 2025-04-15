from django.db import models
from django.utils.translation import gettext_lazy as _


class Event_Source(models.TextChoices):
    AGENT = 'agent', _('Agent')
    MEMORY = 'memory', _('Memory')
    EYE = 'eye', _('Eye')
    BRAIN = 'brain', _('Brain')
    PERCEPTION = 'perception', _('Perception')
    NEURON = 'neuron', _('Neuron')
    DETECT = 'neuron_detect', _('Detect')
    CLASSIFY = 'neuron_classify', _('Classify')
