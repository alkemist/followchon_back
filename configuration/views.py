from datetime import datetime

from django.db.models import Prefetch
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from rest_framework.response import Response

from api.models import ReadOnlyViewSet, CustomPageNumberPagination, UpdateViewSet
from configuration.models import Family, Zone
from configuration.serializers.family import FamilySerializer
from configuration.serializers.family_detections import FamilyDetectionsSerializer
from configuration.serializers.serializers import ZoneSerializer
from detections.models import Detection


class FamilyViewSet(ReadOnlyViewSet):
    queryset = Family.objects.all().order_by('id')
    serializer_class = FamilySerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    pagination_class = CustomPageNumberPagination
    http_method_names = ['get', 'head', 'options']

    def get_queryset(self):
        queryset = Family.objects.all()

        if 'pk' in self.kwargs:

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

        return queryset

    @action(detail=True)
    def detections(self, request, pk=None, *args, **kwargs):
        family = self.get_object()

        return Response(FamilyDetectionsSerializer(family).data)


class ZoneViewSet(UpdateViewSet):
    queryset = Zone.objects.all().order_by('id')
    serializer_class = ZoneSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    pagination_class = CustomPageNumberPagination
    http_method_names = ['get', 'patch', 'head', 'options']
