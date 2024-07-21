from rest_framework.permissions import IsAuthenticatedOrReadOnly

from api.models import ReadOnlyViewSet, CustomPageNumberPagination
from configuration.models import Family, Zone
from configuration.serializers import FamilySerializer, ZoneSerializer


class FamilyViewSet(ReadOnlyViewSet):
    queryset = Family.objects.all().order_by('id')
    serializer_class = FamilySerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    pagination_class = CustomPageNumberPagination


class ZoneViewSet(ReadOnlyViewSet):
    queryset = Zone.objects.all().order_by('id')
    serializer_class = ZoneSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    pagination_class = CustomPageNumberPagination
