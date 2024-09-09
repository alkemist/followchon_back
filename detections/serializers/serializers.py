import pathlib

from rest_framework import serializers

from configuration.models import Family, Zone
from configuration.serializers.family import FamilySerializer
from configuration.serializers.serializers import ZoneSerializer
from detections.models import Capture, Detection
from helpers.yolo import YoloHelper


class DetectionSerializer(serializers.ModelSerializer):
    family = FamilySerializer()
    zone = ZoneSerializer()
    coords = serializers.DictField()
    family_id = serializers.IntegerField()
    zone_id = serializers.IntegerField()

    # family_id = serializers.PrimaryKeyRelatedField(queryset=Family.objects.all())
    # zone_id = serializers.PrimaryKeyRelatedField(queryset=Zone.objects.all())

    class Meta:
        model = Detection
        fields = ['id', 'family', 'zone', 'score', 'family_id', 'zone_id', 'coords', 'trigger']


class CaptureHydratedSerializer(serializers.ModelSerializer):
    detections = DetectionSerializer(many=True)

    detections_ids = serializers.PrimaryKeyRelatedField(many=True, queryset=Detection.objects.all())

    class Meta:
        model = Capture
        fields = ['id', 'date', 'status', 'detections', 'detections_ids', 'photo_path', 'size']

    def update(self, instance, validated_data):
        zones = Zone.objects.all().filter(is_enabled=True)
        lines = list()

        for detection in instance.detections.all():
            detection.delete()

        for index, detection in enumerate(validated_data.get('detections')):
            family_id = detection.get('family_id')

            new_detection = Detection()

            new_detection.capture = instance
            new_detection.family = Family.objects.get(pk=family_id)
            new_detection.family_id = detection.get('family_id')

            coords = detection.get('coords')

            size = instance.size()
            yolo_points = YoloHelper.calc_yolo_points(
                coords['tl_x'], coords['tl_y'], coords['br_x'], coords['br_y'],
                size[0], size[1]
            )

            if new_detection.family.is_zoned:
                for zone in zones:
                    if zone.has_point((yolo_points['x_center'], yolo_points['y_center'])):
                        new_detection.zone = zone
                        break

            new_detection.score = detection.get('score')

            new_detection.center_x = yolo_points['x_center']
            new_detection.center_y = yolo_points['y_center']
            new_detection.width = yolo_points['width']
            new_detection.height = yolo_points['height']

            lines.append(
                (f"{new_detection.family.index} " +
                 f"{yolo_points['x_center']} {yolo_points['y_center']} " +
                 f"{yolo_points['width']} {yolo_points['height']}")
            )

            new_detection.save()

        file = pathlib.Path(instance.label_path(None, True))
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("\n".join(lines))

        if instance.status == Capture.Statuses.DRAFT:
            instance.mark_as(Capture.Statuses.VERIFIED, True)
            instance.status = Capture.Statuses.VERIFIED

        instance.save()

        return instance


class CaptureDateSerializer(serializers.ModelSerializer):
    date_only = serializers.DateField(format="%Y-%m-%d %w")

    class Meta:
        model = Capture
        fields = ['date_only']
