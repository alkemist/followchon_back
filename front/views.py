import json

from detections.models import Capture
from django.db.models import Min, Max
from django.shortcuts import render
from django.views import generic


def index(request):

    return render(request, "index.html", {})
