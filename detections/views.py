from django.db.models import Min, Max
from rest_framework.decorators import action
from rest_framework.response import Response

from api.models import UpdateViewSet
from detections.models import Capture
from detections.serializers import CaptureSerializer, CaptureHydratedSerializer
from src.helpers.math import Math


class CaptureViewSet(UpdateViewSet):
    queryset = Capture.objects.all()
    serializer_class = CaptureHydratedSerializer
    permission_classes = []
    http_method_names = ['get', 'patch', 'head', 'options']

    def get_queryset(self):
        queryset = Capture.objects.all().order_by('-date')

        status = self.request.query_params.get('status')

        if status is not None:
            queryset = queryset.filter(status=status)
        elif 'pk' not in self.kwargs:
            queryset = queryset.filter(status=Capture.DRAFT)

        return queryset

    @action(detail=True)
    def mark_as_draft(self, request, pk=None, *args, **kwargs):
        capture = self.get_object()

        if capture.status != Capture.DRAFT:
            capture.mark_as(Capture.DRAFT, True)
            capture.status = Capture.DRAFT
            capture.save()

        return Response(CaptureHydratedSerializer(capture).data)

    @action(detail=True)
    def mark_as_verified(self, request, pk=None, *args, **kwargs):
        capture = self.get_object()

        if capture.status != Capture.VERIFIED:
            capture.mark_as(Capture.VERIFIED, True)
            capture.status = Capture.VERIFIED
            capture.save()

        return Response(CaptureHydratedSerializer(capture).data)

    @action(detail=True)
    def mark_as_archived(self, request, pk=None, *args, **kwargs):
        capture = self.get_object()

        if capture.status != Capture.ARCHIVED:
            capture.mark_as(Capture.ARCHIVED, True)
            capture.status = Capture.ARCHIVED
            capture.save()

        return Response(CaptureHydratedSerializer(capture).data)

    @action(detail=False)
    def carousel(self, request, *args, **kwargs):
        by_page = 3

        min_id = Capture.objects.aggregate(Min('id')).get('id__min')
        max_id = Capture.objects.aggregate(Max('id')).get('id__max')
        current_id = max_id if not request.GET.get("id") else int(request.GET.get("id"))

        query_min_id, query_max_id = Math.determine_id_range(current_id, min_id, max_id, by_page)

        capture = Capture.objects \
            .prefetch_related('detections') \
            .get(pk=current_id)

        captures = \
            Capture.objects \
                .filter(id__gte=query_min_id, id__lte=query_max_id) \
                .prefetch_related('detections') \
                .order_by("-date")

        captures = list(captures)

        has_pages = len(captures) > 1

        prev_id = current_id + 1 if has_pages and current_id < max_id else False
        next_id = current_id - 1 if has_pages and current_id > min_id else False

        return Response({
            "capture": CaptureHydratedSerializer(capture).data,
            "captures": CaptureSerializer(captures, many=True).data,
            "has_items": len(captures) > 0,
            "current_id": current_id,
            "prev_id": prev_id,
            "next_id": next_id,
        })
