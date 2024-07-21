from django.db.models import Q
from rest_framework.decorators import action
from rest_framework.response import Response

from api.models import UpdateViewSet, CustomLimitOffsetPagination
from detections.models import Capture
from detections.serializers import CaptureHydratedSerializer


class CaptureViewSet(UpdateViewSet):
    queryset = Capture.objects.all()
    serializer_class = CaptureHydratedSerializer
    # permission_classes = [IsAuthenticated, IsAdminUser]
    pagination_class = CustomLimitOffsetPagination
    http_method_names = ['get', 'patch', 'head', 'options']

    def get_queryset(self):
        queryset = Capture.objects.all().order_by('-date')

        id = self.request.query_params.get('id')
        status = self.request.query_params.get('status')
        offset = self.request.query_params.get('offset')
        limit = self.request.query_params.get('limit')

        if id is not None and id:
            if int(offset) == 0 and int(limit) == 1:
                queryset = queryset.filter(id=id)
            else:
                queryset = queryset.filter(~Q(id=id))

        if status is not None and status:
            queryset = queryset.filter(status=status)
        elif 'pk' not in self.kwargs:
            queryset = queryset.filter(status=Capture.Statuses.DRAFT)

        return queryset

    @action(detail=True)
    def mark_as_draft(self, request, pk=None, *args, **kwargs):
        capture = self.get_object()

        if capture.status != Capture.Statuses.DRAFT:
            capture.mark_as(Capture.Statuses.DRAFT, True)
            capture.status = Capture.Statuses.DRAFT
            capture.save()

        return Response(CaptureHydratedSerializer(capture).data)

    @action(detail=True)
    def mark_as_verified(self, request, pk=None, *args, **kwargs):
        capture = self.get_object()

        if capture.status != Capture.Statuses.VERIFIED:
            capture.mark_as(Capture.Statuses.VERIFIED, True)
            capture.status = Capture.Statuses.VERIFIED
            capture.save()

        return Response(CaptureHydratedSerializer(capture).data)

    @action(detail=True)
    def mark_as_archived(self, request, pk=None, *args, **kwargs):
        capture = self.get_object()

        if capture.status != Capture.Statuses.ARCHIVED:
            capture.mark_as(Capture.Statuses.ARCHIVED, True)
            capture.status = Capture.Statuses.ARCHIVED
            capture.save()

        return Response(CaptureHydratedSerializer(capture).data)
