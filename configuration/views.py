import sys

from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from rest_framework.response import Response

from api.models import ReadOnlyViewSet, CustomPageNumberPagination, UpdateViewSet
from configuration.models import Family, Zone
from configuration.serializers.family import FamilySerializer
from configuration.serializers.serializers import ZoneSerializer


class FamilyViewSet(ReadOnlyViewSet):
    queryset = Family.objects.all().order_by('id')
    serializer_class = FamilySerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    pagination_class = CustomPageNumberPagination
    http_method_names = ['get', 'head', 'options']

    def get_queryset(self):
        queryset = Family.objects.all()
        # # queryset = queryset.filter(Q(detections__capture__=Capture.Statuses.DRAFT)
        # #                            | Q(detections__capture__=Capture.Statuses.VERIFIED)
        # #                            | Q(detections__capture__=Capture.Statuses.ARCHIVED))
        # queryset = queryset
        # queryset = queryset.order_by('detections__id')
        return queryset

    @action(detail=True)
    def detections(self, request, pk=None, *args, **kwargs):
        print("pk", pk, file=sys.stderr)
        print("kwargs", self.kwargs, file=sys.stderr)

        family = self.get_object()
        # family = (self.queryset
        #           .filter(id=pk)
        #           .filter(detections__capture__date__range=["2024-08-07 00:00", "2024-08-07 23:59"])
        #           .get(pk=pk))
        # detections = Detection.objects.all() \
        #     .filter(capture__date__range=["2024-08-07 00:00", "2024-08-07 23:59"])

        # return Response(FamilyDetectionsSerializer(family).data)
        return Response(FamilySerializer(family).data)
        # return Response(pk)


class ZoneViewSet(UpdateViewSet):
    queryset = Zone.objects.all().order_by('id')
    serializer_class = ZoneSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    pagination_class = CustomPageNumberPagination
    http_method_names = ['get', 'patch', 'head', 'options']
