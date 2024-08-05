from rest_framework import viewsets, mixins
from rest_framework.pagination import PageNumberPagination, LimitOffsetPagination


class ReadOnlyViewSet(mixins.RetrieveModelMixin,
                      mixins.ListModelMixin,
                      viewsets.GenericViewSet):
    # Replace rest_framework.viewsets.ModelViewSet
    pass


class UpdateViewSet(mixins.RetrieveModelMixin,
                    mixins.ListModelMixin,
                    mixins.UpdateModelMixin,
                    viewsets.GenericViewSet):
    # Replace rest_framework.viewsets.ModelViewSet
    pass


class CustomPageNumberPagination(PageNumberPagination):
    page_size = 100
    page_size_query_param = 'page_size'
    max_page_size = 10000


class CustomLimitOffsetPagination(LimitOffsetPagination):
    limit_query_param = 'limit'
    offset_query_param = 'offset'
