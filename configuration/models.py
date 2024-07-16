from django.db import models


class Family(models.Model):
    index = models.IntegerField(null=True)
    name = models.CharField(max_length=200)
    tracked = models.BooleanField(default=False)
    trigger = models.BooleanField(default=False)
    parent = models.ForeignKey('Family', on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return self.name


class Rule(models.Model):
    score_min = models.IntegerField(default=60)
    enabled = models.BooleanField(default=False)


class Zone(models.Model):
    slug = models.CharField(max_length=200)
    name = models.CharField(max_length=200)

    center_x = models.FloatField(null=True, default=0)
    center_y = models.FloatField(null=True, default=0)
    width = models.FloatField(null=True, default=0)
    height = models.FloatField(null=True, default=0)

    trigger = models.BooleanField(default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.point_tl = (
            self.center_x - self.width / 2,
            self.center_y - self.height / 2
        )
        self.point_br = (
            self.center_x + self.width / 2,
            self.center_y + self.height / 2
        )

    def has_point(self, point):
        has_in_x = self.point_tl[0] <= point[0] <= self.point_br[0]
        has_in_y = self.point_tl[1] <= point[1] <= self.point_br[1]
        return has_in_x and has_in_y

    def __str__(self):
        return self.name
