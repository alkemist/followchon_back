from django.urls import include, path
from rest_framework import routers
from rest_framework.authtoken import views

from configuration import views as views_configuration
from detections import views as views_detections

router = routers.DefaultRouter()

router.register(r'families', views_configuration.FamilyViewSet)
router.register(r'zones', views_configuration.ZoneViewSet)
router.register(r'parameters', views_configuration.ParameterViewSet)
router.register(r'captures', views_detections.CaptureViewSet)
router.register(r'detections', views_detections.DetectionViewSet)

app_name = "api"
urlpatterns = [
    path('api-token-auth/', views.obtain_auth_token),
    path('api-auth/', include('rest_framework.urls')),
    path('api/', include(router.urls)),
]
