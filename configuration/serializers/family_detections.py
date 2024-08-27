from rest_framework import serializers

from configuration.models import Family
from configuration.serializers.serializers import FamilyParentSerializer
from detections.serializers.detection_family import DetectionFamilySerializer


class FamilyDetectionsSerializer(serializers.ModelSerializer):
    parent_id = serializers.PrimaryKeyRelatedField(queryset=Family.objects.all())
    parent = FamilyParentSerializer()

    detections = DetectionFamilySerializer(many=True, read_only=True)

    class Meta:
        model = Family
        fields = [
            'id', 'index', 'name', 'color', 'parent_id', 'parent',
            'is_tracked', 'is_trigger', 'is_abstract', 'is_unique', 'is_zoned',
            'detections'
        ]
