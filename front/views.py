import json

from detections.models import Capture
from django.db.models import Min, Max
from django.shortcuts import render
from django.views import generic


def determine_id_range(id_central, id_min, id_max, x):
    # Calculer le nombre d'éléments de chaque côté de l'id_central
    half_range = (x - 1) // 2

    # Initialiser les limites de la plage autour de id_central
    start_id = id_central - half_range
    end_id = id_central + half_range

    # Ajuster les limites si elles sortent des bornes définies par id_min et id_max
    if start_id < id_min:
        start_id = id_min
        end_id = min(id_min + x - 1, id_max)  # Assurer un maximum de x éléments
    if end_id > id_max:
        end_id = id_max
        start_id = max(id_max - x + 1, id_min)  # Assurer un maximum de x éléments

    return start_id, end_id


def index(request):
    by_page = 3

    min_id = Capture.objects.aggregate(Min('id')).get('id__min')
    max_id = Capture.objects.aggregate(Max('id')).get('id__max')
    current_id = max_id if not request.GET.get("id") else int(request.GET.get("id"))

    query_min_id, query_max_id = determine_id_range(current_id, min_id, max_id, by_page)

    capture = Capture.objects \
        .prefetch_related('detections') \
        .get(pk=current_id)

    detections = [d.toJson() for d in capture.detections.all()]

    captures = \
        Capture.objects \
            .filter(id__gte=query_min_id, id__lte=query_max_id) \
            .prefetch_related('detections') \
            .order_by("-date")

    captures = list(captures)

    has_pages = len(captures) > 1

    prev_id = current_id + 1 if has_pages and current_id < max_id else False
    next_id = current_id - 1 if has_pages and current_id > min_id else False

    print(capture.size())

    return render(request, "index.html", {
        "detections": json.dumps(detections),
        "size": capture.size(),
        "captures": captures,
        "has_items": len(captures) > 0,
        "current_id": current_id,
        "prev_id": prev_id,
        "next_id": next_id,
    })


class DetailView(generic.DetailView):
    model = Capture
    template_name = "capture.html"
