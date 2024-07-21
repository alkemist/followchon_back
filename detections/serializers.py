import pathlib

from rest_framework import serializers

from configuration.models import Family, Zone
from configuration.serializers import FamilySerializer, ZoneSerializer
from detections.models import Capture, Detection
from helpers.yolo import YoloHelper


class DetectionSerializer(serializers.ModelSerializer):
    family_id = serializers.PrimaryKeyRelatedField(queryset=Family.objects.all())
    zone_id = serializers.PrimaryKeyRelatedField(queryset=Zone.objects.all())
    family = FamilySerializer()
    zone = ZoneSerializer()

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
        detections_coords = [d.get('coords') for d in self.data.get('detections')]
        lines = list()

        for detection in instance.detections.all():
            detection.delete()

        for index, detection in enumerate(validated_data.get('detections')):
            family_index = detection.get('family').get('index')
            zone_slug = detection.get('zone').get('slug')

            new_detection = Detection()

            new_detection.capture = instance
            new_detection.family = Family.objects.get(index=family_index)
            new_detection.zone = Zone.objects.get(index=zone_slug)

            new_detection.score = detection.get('score')

            coords = detections_coords[index]

            size = instance.size()
            yolo_points = YoloHelper.calc_yolo_points(
                coords['x1'], coords['y1'], coords['x2'], coords['y2'],
                size[0], size[1]
            )

            new_detection.center_x = yolo_points['x_center']
            new_detection.center_y = yolo_points['y_center']
            new_detection.width = yolo_points['w']
            new_detection.height = yolo_points['h']

            lines.append(
                (f"{family_index} {yolo_points['x_center']} {yolo_points['y_center']} " +
                 f"{yolo_points['w']} {yolo_points['h']}")
            )

            new_detection.save()

        file = pathlib.Path(instance.label_path(None, True))
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("\n".join(lines))

        instance.save()

        return instance


class CaptureSerializer(serializers.ModelSerializer):
    class Meta:
        model = Capture
        fields = ['id', 'photo_path']
