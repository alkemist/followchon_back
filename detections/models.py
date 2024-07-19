import os
import pathlib
import shutil
from typing import cast

import cv2
from django.db import models
from django.utils import timezone
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _

from configuration.models import Family, Zone
from detections.management.commands.vision_models.annotation import Annotation
from helpers.image import ImageHelper
from helpers.yolo import YoloHelper


class Capture(models.Model):
    class Statuses(models.TextChoices):
        DRAFT = 'draft', _('Draft')
        VERIFIED = 'verified', _('Verified')
        ARCHIVED = 'archived', _('Archived')

    base_dir = 'static'
    static_dir = 'captures'
    images_dir = 'images'
    labels_dir = 'labels'

    zones = Zone.objects.all()

    status = models.CharField(null=True, max_length=200, choices=Statuses.choices, default=Statuses.DRAFT)

    photo_file = models.CharField(null=True, max_length=200)

    date = models.DateTimeField(default=timezone.now)

    detections = []

    def detections_ids(self):
        return self.detections

    def write(self, frame: cv2.typing.MatLike, capture_width: float, capture_height: float,
              annotations: list[Annotation]):
        self.status = Capture.Statuses.DRAFT
        self.date = timezone.now()
        self.photo_file = f"{self.date.strftime('%Y-%m-%d_%H-%M-%S-%f')}.jpg"

        photo_dir = f"{self.file_dir(None, True)}{self.images_dir}"

        if not os.path.exists(photo_dir):
            os.makedirs(photo_dir)

        cv2.imwrite(
            self.photo_path(None, True),
            ImageHelper.resize_with_ratio(frame, capture_width, capture_height)
        )

        self.save()

        for annotation in annotations:
            detection = Detection()
            detection.capture = self

            detection.center_x = annotation.yolo_points['x_center']
            detection.center_y = annotation.yolo_points['y_center']
            detection.width = annotation.yolo_points['w']
            detection.height = annotation.yolo_points['h']
            detection.score = annotation.score
            detection.zone = annotation.zone
            detection.trigger = annotation.trigger

            detection.family = annotation.family

            detection.save()

        file = pathlib.Path(self.label_path(None, True))
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("\n".join([annotation.line for annotation in annotations]))

    def size(self):
        im = cv2.imread(self.photo_path(None, True))
        return im.shape[1::-1]

    def mark_as(self, new_status: str, root: bool = None):
        if new_status not in self.STATUSES.keys():
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

    def photo_path(self, status: str = None, root: bool = None):
        return (f"{self.file_dir(status, root)}"
                f"{self.images_dir}/{self.photo_file}")

    def label_path(self, status: str = None, root: bool = None):
        capture_name = pathlib.Path(f"{self.photo_file}").stem
        return (f"{self.file_dir(status, root)}"
                f"{self.labels_dir}/{capture_name}.txt")

    def image_tag(self):
        return mark_safe('<a href="/%s" target="_blank">'
                         '<img src="/%s" width="150" height="150" />'
                         '</a>' % (self.photo_path(), self.photo_path()))

    image_tag.short_description = 'Image'

    def __str__(self):
        return f"{self.date}"


class Detection(models.Model):
    class Triggers(models.TextChoices):
        MOVE = 'move', _('Move')
        ZONE = 'zone', _('Zone')
        FAMILY = 'family', _('Family')

    capture = models.ForeignKey(Capture, on_delete=models.CASCADE, related_name='detections')
    family = models.ForeignKey(Family, on_delete=models.RESTRICT)
    zone = models.ForeignKey(Zone, on_delete=models.SET_NULL, null=True)

    center_x = models.FloatField(null=True)
    center_y = models.FloatField(null=True)
    width = models.FloatField(null=True)
    height = models.FloatField(null=True)

    score = models.FloatField(null=True)
    trigger = models.CharField(null=True, max_length=100, choices=Triggers.choices)

    coords = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}

    def __str__(self):
        return f"{self.family.name}{" in " + self.zone.name if self.zone else ''}"

    def size(self) -> tuple[int, int]:
        return cast(self.capture, Capture).size()

    def coords(self):
        size = self.size()
        return YoloHelper.calc_orthogonal_points(
            x_center_norm=self.center_x,
            y_center_norm=self.center_y,
            w_norm=self.width,
            h_norm=self.height,
            w_img=size[0],
            h_img=size[1],
        )
