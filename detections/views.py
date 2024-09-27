from datetime import datetime

from django.db.models import Q, Count, Case, When, IntegerField, ExpressionWrapper, F, FloatField
from django.db.models.functions import TruncDate, Round
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.response import Response

from api.models import UpdateViewSet, CustomLimitOffsetPagination
from detections.models import Capture
from detections.serializers.serializers import CaptureHydratedSerializer, CaptureDateSerializer, \
    CaptureStatisticsDaySerializer


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
                if sort_value == 'desc':
                    sort_field = '-' + sort_field

                queryset = queryset.order_by(sort_field)
            else:
                queryset = queryset.order_by('-date')

        return queryset

    @action(detail=False)
    def dates(self, request, *args, **kwargs):
        captures = (Capture.objects.all()
                    .annotate(date_only=TruncDate('date'))
                    .values('date_only').distinct()
                    .order_by('-date_only'))

        return Response(CaptureDateSerializer(captures, many=True).data)

    @action(detail=False)
    def statistics_by_day(self, request, *args, **kwargs):
        captures = (
            Capture.objects.all()
            .filter(date__gte=datetime(2024, 9, 4))
            .annotate(
                date_only=TruncDate('date'),
            )
            .values('date_only').distinct()
            .annotate(
                capture_count=Count('id'),
                capture_changed_count=Count(
                    Case(
                        When(changed=True, then=1),
                        output_field=IntegerField(),
                    )
                ),
                capture_changed_percent=ExpressionWrapper(
                    Round(
                        F('capture_changed_count') * 100.0 / F('capture_count'),
                        1
                    ),
                    output_field=FloatField()
                )
            )
            .order_by('date_only')
        )

        return Response(CaptureStatisticsDaySerializer(captures, many=True).data)

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
