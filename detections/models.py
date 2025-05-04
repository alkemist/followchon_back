import os
import pathlib
import shutil
from datetime import datetime
from typing import cast

import cv2
from django.contrib import admin
from django.db import models
from django.utils import timezone
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
from dotenv import load_dotenv

from configuration.models import Family, Zone
from detections.management.commands.models.enums.agent_source import Agent_Source
from detections.management.commands.models.signal import Signal
from utils.image import ImageHelper
from utils.yolo import YoloHelper

load_dotenv()


class Capture(models.Model):
    class Statuses(models.TextChoices):
        DRAFT = 'draft', _('Draft')
        VERIFIED = 'verified', _('Verified')
        ARCHIVED = 'archived', _('Archived')
        WAITING = 'waiting', _('Waiting')
        DELETED = 'deleted', _('Deleted')

    STATUS_EDITABLE = 'editable'
    STATUS_ALL = 'all'
    STATUS_ERROR = 'error'

    base_dir = 'static'
    static_dir = 'captures'
    images_dir = 'images'
    labels_dir = 'labels'

    status = models.CharField(null=True, max_length=200, choices=Statuses.choices, default=Statuses.DRAFT)
    source = models.CharField(null=True, max_length=200, choices=Agent_Source.choices, default=Agent_Source.VISION)

    photo_file = models.CharField(null=True, max_length=200)

    date = models.DateTimeField(default=timezone.now)
    changed = models.BooleanField(default=False)
    version_detect = models.IntegerField(null=True, default=None)
    version_classify = models.IntegerField(null=True, default=None)
    errors = models.IntegerField(null=True, default=None)
    train_all = models.BooleanField(default=False)
    train_chons = models.BooleanField(default=False)

    id = 0
    detections = []

    def detections_ids(self):
        return self.detections

    def write(self, frame: cv2.typing.MatLike, date: datetime, annotations: list[Signal],
              version_detect: int,
              version_classify: int,
              source: Agent_Source,
              errors: int = 0):
        self.status = Capture.Statuses.DRAFT
        self.date = date
        self.photo_file = f"{self.date.strftime('%Y-%m-%d_%H-%M-%S-%f')}.jpg"
        self.changed = False
        self.version_detect = version_detect
        self.version_classify = version_classify
        self.source = source
        self.errors = errors

        photo_dir = f"{self.file_dir(None, True)}{self.images_dir}"

        if not os.path.exists(photo_dir):
            os.makedirs(photo_dir)

        self.save_image(frame)
        self.save()

        for annotation in annotations:
            detection = Detection()
            detection.capture = self

            detection.center_x = annotation.norm_x_center
            detection.center_y = annotation.norm_y_center
            detection.width = annotation.norm_width
            detection.height = annotation.norm_height
            detection.score = annotation.score
            detection.family = annotation.family
            detection.zone = annotation.zone
            detection.trigger = annotation.trigger

            detection.save()

        file = pathlib.Path(self.label_path(None, True))
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("\n".join([annotation.line for annotation in annotations]))

    def check_changed(self):
        detections = Detection.objects.filter(capture_id=self.id)
        self.changed = any([d.score is None or d.score == 0 for d in detections])
        self.save()

    def calc_errors(self):
        detections_guineapig = Detection.objects.filter(capture_id=self.id, family_id=1)
        detections_chons = Detection.objects.filter(capture_id=self.id).exclude(family_id=1)

        if self.errors is None:
            self.errors = 0

            # @TODO Rendre les valeurs dynamiques
            if len(detections_guineapig) > 2:
                self.errors = 3
            elif len(detections_guineapig) > len(detections_chons):
                self.errors = 2
            if len(detections_guineapig) == 1:
                self.errors = 1

            self.save()

    def save_image(self, image):
        cv2.imwrite(
            self.photo_path(None, True),
            ImageHelper.resize_with_ratio(image, int(os.getenv('CAPTURE_WIDTH')))
        )

    def resize_auto(self):
        image = cv2.imread(self.photo_path(None, True))

        self.save_image(image)

    def size(self):
        im = cv2.imread(self.photo_path(None, True))
        return im.shape[1::-1] if im is not None else (0, 0)

    def mark_as(self, new_status: str, root: bool = None):
        if new_status not in Capture.Statuses or not os.path.isfile(self.photo_path(self.status, root)):
            return

        shutil.move(self.photo_path(self.status, root), self.photo_path(new_status, root))
        shutil.move(self.label_path(self.status, root), self.label_path(new_status, root))

    def remove_files(self):
        image_path = self.photo_path(None, True)
        label_path = self.label_path(None, True)

        if os.path.isfile(image_path):
            os.remove(image_path)
        if os.path.isfile(label_path):
            os.remove(label_path)

    def file_dir(self, status: str | None = None, root: bool = None):
        status = self.status if status is None else status

        if status not in Capture.Statuses:
            return ""

        return (f"{self.base_dir + '/' if root is not None and root else ''}"
                f"{self.static_dir}/"
                f"{status}/"
                )

    def photo_path(self, status: str = None, root: bool = True):
        return (f"{self.file_dir(status, root)}"
                f"{self.images_dir}/{self.photo_file}")

    def label_path(self, status: str = None, root: bool = None):
        capture_name = pathlib.Path(f"{self.photo_file}").stem
        return (f"{self.file_dir(status, root)}"
                f"{self.labels_dir}/{capture_name}.txt")

    @admin.display(description='photo')
    def image_tag(self):
        return mark_safe('<a href="/%s" target="_blank">'
                         '<img src="/%s" width="200" height="200" />'
                         '</a>' % (self.photo_path(root=True), self.photo_path(root=True)))

    def front_url(self):
        return format_html(
            '<a target="_blank" href="{0}">{1}</a>',
            f"{os.getenv('FRONT_URL')}?id={self.id}&status={self.status}",
            'Followchon',
        )

    def __str__(self):
        return f"{self.date.strftime('%d %B %Y %H:%M:%S %f')}"


class Detection(models.Model):
    class Triggers(models.TextChoices):
        MOVE = 'move', _('Move')
        ZONE = 'zone', _('Zone')
        FAMILY = 'family', _('Family')

    capture = models.ForeignKey(Capture, on_delete=models.CASCADE, related_name='detections')
    family = models.ForeignKey(Family, on_delete=models.RESTRICT, related_name='detections')
    zone = models.ForeignKey(Zone, on_delete=models.SET_NULL, null=True, default=None, blank=True)

    center_x = models.FloatField(null=True, default=0)
    center_y = models.FloatField(null=True, default=0)
    width = models.FloatField(null=True, default=0)
    height = models.FloatField(null=True, default=0)

    score = models.FloatField(null=True, default=0, blank=True)
    trigger = models.CharField(null=True, max_length=100, choices=Triggers.choices, default=None, blank=True)

    coords = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}

    def size(self) -> tuple[int, int]:
        return cast(self.capture, Capture).size(self.capture)

    def image_tag(self):
        return mark_safe(
            '<a href="/%s" target="_blank">'
            '<img src="/%s" width="200" height="200" />'
            '</a>' % (cast(self.capture, Capture).photo_path(self.capture),
                      cast(self.capture, Capture).photo_path(self.capture))
        )

    def line(self):
        return (f"{self.family.index} {self.center_x} {self.center_y} " +
                f"{self.width} {self.height}")

    def coords(self):
        size = self.size()
        if size[0] == 0 or size[1] == 0:
            return {
                'tl_x': 0,
                'tl_y': 0,
                'br_x': 0,
                'br_y': 0,
            }

        return YoloHelper.calc_orthogonal_points(
            x_center_norm=self.center_x,
            y_center_norm=self.center_y,
            w_norm=self.width,
            h_norm=self.height,
            w_img=size[0],
            h_img=size[1],
        )
