from rest_framework.routers import DefaultRouter
from .views import MLTaskViewSet

router = DefaultRouter()
router.register(r'ml-tasks', MLTaskViewSet, basename='mltask')

urlpatterns = router.urls
