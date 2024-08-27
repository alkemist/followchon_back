import datetime

from rest_framework import serializers

from configuration.serializers.serializers import ZoneSerializer
from detections.models import Detection

datetime.datetime.now(tz=datetime.timezone.utc)


class DetectionFamilySerializer(serializers.ModelSerializer):
    zone = ZoneSerializer()
    zone_id = serializers.IntegerField()
    date = serializers.SerializerMethodField()
    status = serializers.SerializerMethodField()
    photo_path = serializers.SerializerMethodField()
    capture_id = serializers.SerializerMethodField()

    def get_date(self, obj):
        return obj.capture.date

    def get_status(self, obj):
        return obj.capture.status

    def get_photo_path(self, obj):
        return obj.capture.photo_path()

    def get_capture_id(self, obj):
        return obj.capture.id

    class Meta:
        model = Detection
        fields = ['id', 'zone', 'score', 'zone_id', 'trigger', 'date', 'status', 'photo_path',
                  'capture_id', ]
