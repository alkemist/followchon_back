from django.urls import path

from front import views

app_name = "front"
urlpatterns = [
    path("", views.index, name="index"),
]
