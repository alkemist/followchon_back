import pathlib
import shutil

import cv2
from configuration.models import Family, Zone
from django.db import models
from django.utils.safestring import mark_safe

from src.helpers.yolo import YoloHelper


class Capture(models.Model):
    base_dir = 'static/captures'
    images_dir = 'images'
    labels_dir = 'labels'
    verified_dir = 'verified'
    draft_dir = 'draft'

    verified = models.BooleanField(default=False)

    photo_file = models.CharField(null=True, max_length=200)

    date = models.DateTimeField()

    def size(self):
        im = cv2.imread(self.photo_path())
        return im.shape[1::-1]

    def move_directory(self, verified):
        shutil.move(self.photo_path(not verified), self.photo_path(verified))
        shutil.move(self.label_path(not verified), self.label_path(verified))

    def photo_path(self, verified=None):
        return (f"{self.base_dir}/"
                f"{self.verified_dir if verified is None and self.verified or verified else self.draft_dir}/"
                f"{self.images_dir}/{self.photo_file}")

    def label_path(self, verified=None):
        capture_name = pathlib.Path(f"{self.photo_file}").stem
        return (f"{self.base_dir}/"
                f"{self.verified_dir if verified is None and self.verified or verified else self.draft_dir}/"
                f"{self.labels_dir}/{capture_name}.txt")

    def image_tag(self):
        return mark_safe('<a href="/%s" target="_blank">'
                         '<img src="/%s" width="150" height="150" />'
                         '</a>' % (self.photo_path(), self.photo_path()))

    image_tag.short_description = 'Image'

    def __str__(self):
        return f"{self.date}"


class Detection(models.Model):
    capture = models.ForeignKey(Capture, on_delete=models.CASCADE, related_name='detections')
    family = models.ForeignKey(Family, on_delete=models.CASCADE)
    zone = models.ForeignKey(Zone, on_delete=models.CASCADE)

    center_x = models.FloatField(null=True, default=0)
    center_y = models.FloatField(null=True, default=0)
    width = models.FloatField(null=True, default=0)
    height = models.FloatField(null=True, default=0)

    score = models.IntegerField(null=True, default=0)

    def __str__(self):
        return f"{self.family.name} in {self.zone.name}"

    def toJson(self):
        w_img, h_img = self.capture.size()
        return {
            'coords': YoloHelper.calc_orthogonal_points(
                x_center_norm=self.center_x,
                y_center_norm=self.center_y,
                w_norm=self.width,
                h_norm=self.height,
                w_img=w_img,
                h_img=h_img
            ),
            'family': self.family.name,
            'zone': self.zone.name,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'width': self.width,
            'height': self.height,

            'score': self.score,
        }
