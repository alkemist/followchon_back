from rest_framework import serializers

from configuration.serializers.serializers import ZoneSerializer
from detections.models import Detection


class DetectionFamilySerializer(serializers.ModelSerializer):
    zone = ZoneSerializer()
    zone_id = serializers.IntegerField()

    class Meta:
        model = Detection
        fields = ['id', 'zone', 'score', 'zone_id', 'trigger']
