from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import HealthView, IngestReceiptView, UploadAndIngestView, ConfirmReceiptView, ReceiptViewSet, JobViewSet

router = DefaultRouter()
router.register(r"receipts", ReceiptViewSet, basename="receipt")
router.register(r"jobs", JobViewSet, basename="job")

urlpatterns = [
    path("healthz/", HealthView.as_view()),
    path("receipt/ingest", IngestReceiptView.as_view()),
    path("receipt/upload", UploadAndIngestView.as_view()),
    path("receipt/<int:receipt_id>/confirm", ConfirmReceiptView.as_view()),
    path("", include(router.urls)),
]
