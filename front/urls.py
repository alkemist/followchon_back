from django.urls import path

from front import views

app_name = "front"
urlpatterns = [
    path("", views.index, name="index"),
    path("detections/<int:pk>", views.DetailView.as_view(), name="detail"),
]
