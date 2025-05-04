from datetime import datetime

from django.db.models import Prefetch, Count
from django.db.models.functions import TruncDate
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from rest_framework.response import Response

from api.models import ReadOnlyViewSet, CustomPageNumberPagination, UpdateViewSet
from configuration.models import Family, Zone, Parameter
from configuration.serializers.family import FamilySerializer
from configuration.serializers.family_detections import FamilyDetectionsSerializer
from configuration.serializers.serializers import ZoneSerializer, ParameterSerializer
from detections.management.commands.models.enums.agent_source import Agent_Source
from detections.models import Detection, Capture
from detections.serializers.detection_family import DetectionCountByDayFamilySerializer, \
    DetectionDistanceByDayFamilySerializer


class ParameterViewSet(ReadOnlyViewSet):
    queryset = Parameter.objects.all().order_by('id')
    serializer_class = ParameterSerializer
    http_method_names = ['get', 'head', 'options']


class FamilyViewSet(ReadOnlyViewSet):
    queryset = Family.objects.all().filter(is_listed=True).order_by('index')
    serializer_class = FamilySerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    pagination_class = CustomPageNumberPagination
    http_method_names = ['get', 'head', 'options']

    def get_queryset(self):
        queryset = Family.objects.all().filter(is_listed=True).order_by('index')

        if 'pk' in self.kwargs:

            if self.request.query_params.get('date'):
                try:
                    date = datetime.strptime(self.request.query_params.get('date'), '%Y-%m-%d')
                except (ValueError, TypeError) as e:
                    date = datetime.now()

                queryset = Family.objects.prefetch_related(
                    Prefetch(
                        'detections',
                        queryset=Detection.objects.filter(
                            capture__source=Agent_Source.VISION,
                            capture__date__range=[
                                date.replace(hour=0, minute=0, second=0),
                                date.replace(hour=23, minute=59, second=59)
                            ])
                        .exclude(capture__status=Capture.Statuses.DELETED)
                        .order_by('capture__date')
                    )
                )
            else:
                queryset = Family.objects.prefetch_related(
                    Prefetch(
                        'detections',
                        queryset=Detection.objects.filter(capture__source=Agent_Source.VISION)
                        .exclude(capture__status=Capture.Statuses.DELETED)
                        .filter(capture__date__gte=datetime(2025, 2, 17))
                    )
                )

        return queryset

    @action(detail=True)
    def detections(self, request, pk=None, *args, **kwargs):
        family = self.get_object()

        return Response(FamilyDetectionsSerializer(family).data)

    @action(detail=True)
    def detections_by_day(self, request, pk=None, *args, **kwargs):
        family = self.get_object()
        detections_count_by_day = (
            family.detections.all()
            .annotate(date_only=TruncDate('capture__date'))
            .values('date_only').distinct()
            .annotate(count=Count('id'))
            .order_by('date_only')
        )

        return Response(DetectionCountByDayFamilySerializer(detections_count_by_day, many=True, read_only=True).data)

    @action(detail=True)
    def distances_by_day(self, request, pk=None, *args, **kwargs):
        family = self.get_object()

        # detections_distance_by_day = (
        #    family.detections.all()
        #    .order_by('capture__date')
        #    .annotate(
        #        center_x_prev=Window(
        #            expression=Lag('center_x'), order_by='capture__date', output_field=FloatField()
        #        ),
        #        center_y_prev=Window(
        #            expression=Lag('center_y'), order_by='capture__date', output_field=FloatField()
        #        ),
        #    )
        #    .annotate(
        #        delta_x=Abs(F('center_x') - F('center_x_prev')),
        #        delta_y=Abs(F('center_y') - F('center_y_prev')),
        #    )
        #    .annotate(
        #        distance=Sqrt(F('delta_x') ** 2 + F('delta_y') ** 2, output_field=FloatField())
        #    )
        #    # .annotate(
        #    #    date_only=TruncDate('capture__date'),
        #    # )
        #    .values('capture__date').distinct()
        #    .annotate(total=Sum('distance'))
        #    # .order_by('date_only')
        # )

        detections_distance_by_day = Detection.objects.raw(f'''
            SELECT 
                date_only AS id, 
                SUM(distance) AS "total"
                FROM (
                    SELECT 
                        date_only,
                        SQRT(POW(delta_x, 2) + POW(delta_y, 2)) AS "distance"
                      FROM (
                               SELECT 
                                    date_only,
                                    ABS(center_x - center_x_prev) AS delta_x,
                                    ABS(center_y - center_y_prev) AS delta_y
                                 FROM (
                                      SELECT 
                                             date(detections_capture."date") AS "date_only",
                                             detections_detection."center_x",
                                             detections_detection."center_y",
                                             LAG(detections_detection."center_x", 1) OVER (ORDER BY detections_capture."date") AS center_x_prev,
                                             LAG(detections_detection."center_y", 1) OVER (ORDER BY detections_capture."date") AS center_y_prev
                                        FROM detections_detection
                                             INNER JOIN detections_capture 
                                             ON (detections_detection."capture_id" = detections_capture."id") 
                                        WHERE detections_capture."source" = "{Agent_Source.VISION}"
                                            AND detections_capture."status" != "{Capture.Statuses.DELETED}"
                                            AND "detections_capture"."date" >= "2025-02-17 00:00:00" 
                                            AND "detections_detection"."family_id" = {family.id}
                                  )
                                  AS prev
                           )
                           AS delta
                ) AS distance
                GROUP BY date_only
        ''')

        return Response(
            DetectionDistanceByDayFamilySerializer(detections_distance_by_day, many=True, read_only=True).data)


class ZoneViewSet(UpdateViewSet):
    queryset = Zone.objects.all().filter(is_enabled=True).order_by('id')
    serializer_class = ZoneSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    pagination_class = CustomPageNumberPagination
    http_method_names = ['get', 'patch', 'head', 'options']
