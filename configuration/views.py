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
from detections.management.commands.vision_models.source import Source
from detections.models import Detection
from detections.serializers.detection_family import DetectionByDayFamilySerializer


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
                            capture__date__range=[
                                date.replace(hour=0, minute=0, second=0),
                                date.replace(hour=23, minute=59, second=59)
                            ])
                        .order_by('capture__date')
                    )
                )
            else:
                queryset = Family.objects.prefetch_related(
                    Prefetch(
                        'detections',
                        queryset=Detection.objects.filter(capture__source=Source.VISION)
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
        detections_by_day = (
            family.detections.all()
            .annotate(date_only=TruncDate('capture__date'))
            .values('date_only').distinct()
            .annotate(count=Count('id'))
            .order_by('date_only')
        )

        return Response(DetectionByDayFamilySerializer(detections_by_day, many=True, read_only=True).data)


class ZoneViewSet(UpdateViewSet):
    queryset = Zone.objects.all().filter(is_enabled=True).order_by('id')
    serializer_class = ZoneSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    pagination_class = CustomPageNumberPagination
    http_method_names = ['get', 'patch', 'head', 'options']
