from rest_framework import viewsets, mixins

from rest_framework.viewsets import ModelViewSet


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
