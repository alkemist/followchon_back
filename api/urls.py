from django.urls import include, path
from rest_framework import routers
from rest_framework.urlpatterns import format_suffix_patterns

from configuration import views as views_configuration
from detections import views as views_detections

router = routers.DefaultRouter()

router.register(r'familys', views_configuration.FamilyViewSet)
router.register(r'zones', views_configuration.ZoneViewSet)
router.register(r'captures', views_detections.CaptureViewSet)

app_name = "api"
urlpatterns = [
    path('api-auth/', include('rest_framework.urls')),
    path('api/', include(router.urls)),
]
