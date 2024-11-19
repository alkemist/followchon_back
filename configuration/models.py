from typing import cast

from django.contrib import admin
from django.db import models
from django.utils import timezone
from django.utils.safestring import mark_safe

from detections.management.commands.vision_models.levels import Levels
from detections.management.commands.vision_models.sources import Sources


class Family(models.Model):
    class Colors(models.TextChoices):
        Coffee = '6f4e37', 'Coffee'
        ChillRed = 'e23e28', 'Chill red'
        Amber = 'ffbf00', 'Amber'
        White = 'ffffff', 'White'

    index = models.IntegerField(null=True)
    name = models.CharField(max_length=200)
    parent = models.ForeignKey('Family', on_delete=models.CASCADE, null=True, blank=True)
    is_tracked = models.BooleanField(default=False)
    is_trigger = models.BooleanField(default=False)
    is_abstract = models.BooleanField(default=False)
    is_unique = models.BooleanField(default=False)
    is_zoned = models.BooleanField(default=False)
    color = models.CharField(null=True, max_length=100, choices=Colors.choices)

    class Meta:
        verbose_name_plural = "families"

    def __str__(self):
        return self.name


class Zone(models.Model):
    slug = models.CharField(max_length=200)
    name = models.CharField(max_length=200)

    center_x = models.FloatField(null=True, default=0)
    center_y = models.FloatField(null=True, default=0)
    width = models.FloatField(null=True, default=0)
    height = models.FloatField(null=True, default=0)

    is_trigger = models.BooleanField(default=False)
    is_ignored = models.BooleanField(default=False)
    is_indoor = models.BooleanField(default=False)
    is_enabled = models.BooleanField(default=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.center_x is not None and self.center_y is not None and self.width is not None and self.height is not None:
            self.norm_point_tl = (
                cast(float, self.center_x) - cast(float, self.width) / 2,
                cast(float, self.center_y) - cast(float, self.height) / 2
            )

            self.norm_point_br = (
                cast(float, self.center_x) + cast(float, self.width) / 2,
                cast(float, self.center_y) + cast(float, self.height) / 2
            )

    def has_point(self, point: tuple[int, int]) -> bool:
        has_in_x = self.norm_point_tl[0] <= point[0] <= self.norm_point_br[0]
        has_in_y = self.norm_point_tl[1] <= point[1] <= self.norm_point_br[1]
        return has_in_x and has_in_y

    def __str__(self):
        return self.name


class Rule(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Period(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Parameter(models.Model):
    slug = models.CharField(max_length=200)
    name = models.CharField(max_length=200, default='', null=True)
    value = models.CharField(max_length=200, default='', null=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return self.name


class Log(models.Model):
    date = models.DateTimeField(default=timezone.now)
    source = models.CharField(null=True, max_length=200, choices=Sources.choices, default=Sources.VISION)
    level = models.CharField(null=True, max_length=200, choices=Levels.choices, default=Levels.INFO)
    event = models.CharField(max_length=200, default='', null=True)
    info = models.CharField(max_length=200, default='', null=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create(self, source: str, level: Levels, event: str, info: str):
        self.source = source
        self.level = level
        self.event = event
        self.info = info
        self.save()

    @admin.display(description='Level')
    def level_color(self):
        color = '#fff'

        match self.level:
            case Levels.EVENT:
                color = '#0ea5e9'
            case Levels.STATISTIC:
                color = '#22c55e'
            case Levels.WARNING:
                color = '#f97316'
            case Levels.HOT:
                color = '#ef4444'
            case Levels.FAIL:
                color = '#a855f7'
            case Levels.ERROR:
                color = '#a855f7'

        return mark_safe(f'<b style="color:{color}">{self.level}</b>')

    def __str__(self):
        return f"{self.date} {self.level} {self.event} {self.info}"
