from django.db.models import Q
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.response import Response

from api.models import UpdateViewSet, CustomLimitOffsetPagination
from detections.models import Capture
from detections.serializers import CaptureHydratedSerializer


class CaptureViewSet(UpdateViewSet):
    queryset = Capture.objects.all()
    serializer_class = CaptureHydratedSerializer
    permission_classes = [IsAuthenticated, IsAdminUser]
    pagination_class = CustomLimitOffsetPagination
    http_method_names = ['get', 'patch', 'head', 'options']

    def get_queryset(self):
        queryset = Capture.objects.all()

        if 'pk' not in self.kwargs:
            capture_id = self.request.query_params.get('id')
            status = self.request.query_params.get('status')
            offset = self.request.query_params.get('offset')
            limit = self.request.query_params.get('limit')
            sort_field = self.request.query_params.get('sortField')
            sort_value = self.request.query_params.get('sortValue')

            if capture_id is not None and capture_id.isdigit():
                if offset is not None and limit is not None and int(offset) == 0 and int(limit) == 1:
                    queryset = queryset.filter(id=capture_id)
                else:
                    queryset = queryset.filter(~Q(id=capture_id))

            if status is not None and status:
                if status == Capture.STATUS_EDITABLE:
                    queryset = queryset.filter(Q(status=Capture.Statuses.DRAFT) | Q(status=Capture.Statuses.VERIFIED))
                elif status == Capture.STATUS_ALL:
                    queryset = queryset.filter(Q(status=Capture.Statuses.DRAFT) | Q(status=Capture.Statuses.VERIFIED)
                                               | Q(status=Capture.Statuses.ARCHIVED))
                else:
                    queryset = queryset.filter(status=status)
            else:
                queryset = queryset.filter(status=Capture.Statuses.DRAFT)

            if sort_field is not None and sort_field and sort_value is not None:
                queryset = queryset.order_by(f'{'-' if sort_value == 'desc' else ''}{sort_field}')
            else:
                queryset = queryset.order_by('-date')

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

    @action(detail=True)
    def mark_as_deleted(self, request, pk=None, *args, **kwargs):
        capture = self.get_object()

        if capture.status != Capture.Statuses.DELETED:
            capture.mark_as(Capture.Statuses.DELETED, True)
            capture.status = Capture.Statuses.DELETED
            capture.save()

        return Response(CaptureHydratedSerializer(capture).data)
