from django.db import models


class Family(models.Model):
    index = models.IntegerField(null=True)
    name = models.CharField(max_length=200)

    def __str__(self):
        return self.name


class Zone(models.Model):
    index = models.CharField(max_length=200)
    name = models.CharField(max_length=200)

    def __str__(self):
        return self.name
