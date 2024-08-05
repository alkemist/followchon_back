from rest_framework.permissions import IsAuthenticatedOrReadOnly

from api.models import ReadOnlyViewSet, CustomPageNumberPagination, UpdateViewSet
from configuration.models import Family, Zone
from configuration.serializers import FamilySerializer, ZoneSerializer


class FamilyViewSet(ReadOnlyViewSet):
    queryset = Family.objects.all().order_by('id')
    serializer_class = FamilySerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    pagination_class = CustomPageNumberPagination
    http_method_names = ['get', 'head', 'options']


class ZoneViewSet(UpdateViewSet):
    queryset = Zone.objects.all().order_by('id')
    serializer_class = ZoneSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    pagination_class = CustomPageNumberPagination
    http_method_names = ['get', 'patch', 'head', 'options']
